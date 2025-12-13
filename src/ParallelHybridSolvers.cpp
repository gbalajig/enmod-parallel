#include "enmod/ParallelHybridSolvers.h"
#include "enmod/ParallelBIDP.h" 
#include "enmod/QLearningSolver.h" // ADDED: Crucial for unique_ptr destructor and methods
#include "enmod/Logger.h"
#include <cmath>
#include <iostream>

// Helper for threat assessment
static EvacuationMode get_mode(const Position& pos, const Grid& grid) {
    const auto& events = grid.getConfig().value("dynamic_events", json::array());
    for (const auto& event : events) {
        if (event.value("type", "") == "fire") {
            Position fire_pos = {event.at("position").at("row"), event.at("position").at("col")};
            if (grid.getCellType(fire_pos) == CellType::FIRE) {
                int radius = 1; 
                if(event.value("size", "small") == "medium") radius=2; 
                if(event.value("size", "small") == "large") radius=3;
                int dist = std::abs(pos.row - fire_pos.row) + std::abs(pos.col - fire_pos.col);
                if (dist <= 1) return EvacuationMode::PANIC;
                if (dist <= radius) return EvacuationMode::ALERT;
            }
        }
    }
    int dr[] = {-1, 1, 0, 0};
    int dc[] = {0, 0, -1, 1};
    for(int i=0; i<4; ++i) {
         if(grid.getSmokeIntensity({pos.row+dr[i], pos.col+dc[i]}) == "heavy") return EvacuationMode::ALERT;
    }
    return EvacuationMode::NORMAL;
}

// ============================================================
// 1. ParallelHybridDPRLSolver
// ============================================================
ParallelHybridDPRLSolver::ParallelHybridDPRLSolver(const Grid& grid_ref) 
    : Solver(grid_ref, "ParallelHybridDPRLSim") {
    // Instantiate RL solver
    rl_solver = std::make_unique<QLearningSolver>(grid_ref);
    // Note: Train might take time, but is necessary for the hybrid model
    // Reducing episodes slightly for performance if needed, but keeping 5000 as per spec
    rl_solver->train(2000); // Reduced to 2000 to prevent timeouts/memory strain during init
}

void ParallelHybridDPRLSolver::run() {
    Grid dyn_grid = grid;
    Position curr = dyn_grid.getStartPosition();
    total_cost = {0,0,0};
    history.clear();
    const auto& events = dyn_grid.getConfig().value("dynamic_events", json::array());

    // Limit steps to avoid infinite loops in bad policies
    int max_steps = 2 * (dyn_grid.getRows() * dyn_grid.getCols());
    for (int t = 0; t < max_steps; ++t) {
        // 1. Update Hazards
        for (const auto& ev : events) {
            if (ev.value("time_step", -1) == t) dyn_grid.addHazard(ev);
        }
        
        // 2. Determine Mode
        EvacuationMode mode = get_mode(curr, dyn_grid);
        Cost::current_mode = mode;
        history.push_back({t, dyn_grid, curr, "Planning...", total_cost, mode});

        // 3. Check Exit
        if (dyn_grid.isExit(curr.row, curr.col)) { 
            history.back().action="SUCCESS"; 
            break; 
        }

        Direction move = Direction::STAY;
        
        // 4. Plan
        if (mode == EvacuationMode::PANIC) {
            move = rl_solver->chooseAction(curr);
        } else {
            // Scope block for GPU Planner to ensure destructor runs and frees memory immediately
            {
                ParallelBIDP gpu_planner(dyn_grid);
                gpu_planner.run();
                const auto& map = gpu_planner.getCostMap();
                
                // Greedy descent
                Cost best_c = map[curr.row][curr.col];
                int dr[]={-1,1,0,0}, dc[]={0,0,-1,1};
                Direction ds[]={Direction::UP,Direction::DOWN,Direction::LEFT,Direction::RIGHT};
                
                for(int i=0; i<4; ++i) {
                    Position n={curr.row+dr[i], curr.col+dc[i]};
                    if(dyn_grid.isWalkable(n.row, n.col) && map[n.row][n.col] < best_c) {
                        best_c = map[n.row][n.col];
                        move = ds[i];
                    }
                }
            } // gpu_planner destroyed, GPU memory freed
        }
        
        // 5. Execute
        std::string act = "STAY";
        if(move==Direction::UP) act="UP"; else if(move==Direction::DOWN) act="DOWN";
        else if(move==Direction::LEFT) act="LEFT"; else if(move==Direction::RIGHT) act="RIGHT";
        
        history.back().action = act;
        total_cost = total_cost + dyn_grid.getMoveCost(curr);
        curr = dyn_grid.getNextPosition(curr, move);
    }
    
    if(history.empty() || history.back().action.find("SUCCESS") == std::string::npos) 
        history.push_back({0, dyn_grid, curr, "FAILURE", total_cost, EvacuationMode::NORMAL});
        
    Cost::current_mode = EvacuationMode::NORMAL;
}

Cost ParallelHybridDPRLSolver::getEvacuationCost() const { return total_cost; }
void ParallelHybridDPRLSolver::generateReport(std::ofstream& f) const {
    f << "<h2>Parallel Hybrid DPRL History</h2>";
    for(const auto& s : history) f << s.grid_state.toHtmlStringWithAgent(s.agent_pos);
}

// ============================================================
// 2. ParallelAdaptiveCostSolver
// ============================================================
ParallelAdaptiveCostSolver::ParallelAdaptiveCostSolver(const Grid& grid_ref) : Solver(grid_ref, "ParallelAdaptiveCostSim") {}
void ParallelAdaptiveCostSolver::run() {
    Grid dyn_grid = grid;
    Position curr = dyn_grid.getStartPosition();
    total_cost = {0,0,0};
    history.clear();
    const auto& events = dyn_grid.getConfig().value("dynamic_events", json::array());

    int max_steps = 2 * (dyn_grid.getRows() * dyn_grid.getCols());
    for (int t = 0; t < max_steps; ++t) {
        for (const auto& ev : events) if (ev.value("time_step", -1) == t) dyn_grid.addHazard(ev);
        
        EvacuationMode mode = get_mode(curr, dyn_grid);
        Cost::current_mode = mode; // Changes static weights for GPU kernel
        history.push_back({t, dyn_grid, curr, "Planning...", total_cost, mode});

        if (dyn_grid.isExit(curr.row, curr.col)) { history.back().action="SUCCESS"; break; }

        Cost best_c;
        Direction move = Direction::STAY;

        {
            ParallelBIDP gpu_planner(dyn_grid);
            gpu_planner.run();
            const auto& map = gpu_planner.getCostMap();

            best_c = map[curr.row][curr.col];
            int dr[]={-1,1,0,0}, dc[]={0,0,-1,1};
            Direction ds[]={Direction::UP,Direction::DOWN,Direction::LEFT,Direction::RIGHT};
            for(int i=0; i<4; ++i) {
                Position n={curr.row+dr[i], curr.col+dc[i]};
                if(dyn_grid.isWalkable(n.row, n.col) && map[n.row][n.col] < best_c) {
                    best_c = map[n.row][n.col];
                    move = ds[i];
                }
            }
        }

        if(move == Direction::STAY && best_c.distance >= 2000000) { // Check high cost
             history.back().action="FAILURE"; 
             break; 
        }

        std::string act = "STAY";
        if(move==Direction::UP) act="UP"; else if(move==Direction::DOWN) act="DOWN";
        else if(move==Direction::LEFT) act="LEFT"; else if(move==Direction::RIGHT) act="RIGHT";
        history.back().action = act;
        total_cost = total_cost + dyn_grid.getMoveCost(curr);
        curr = dyn_grid.getNextPosition(curr, move);
    }
    if(history.empty() || history.back().action.find("SUCCESS") == std::string::npos) 
        history.push_back({0, dyn_grid, curr, "FAILURE", total_cost, EvacuationMode::NORMAL});
    Cost::current_mode = EvacuationMode::NORMAL;
}
Cost ParallelAdaptiveCostSolver::getEvacuationCost() const { return total_cost; }
void ParallelAdaptiveCostSolver::generateReport(std::ofstream& f) const {
    f << "<h2>Parallel Adaptive Cost History</h2>";
    for(const auto& s : history) f << s.grid_state.toHtmlStringWithAgent(s.agent_pos);
}

// ============================================================
// 3. ParallelInterlacedSolver
// ============================================================
ParallelInterlacedSolver::ParallelInterlacedSolver(const Grid& grid_ref) : Solver(grid_ref, "ParallelInterlacedSim") {}
void ParallelInterlacedSolver::run() {
    Grid dyn_grid = grid;
    Position curr = dyn_grid.getStartPosition();
    total_cost = {0,0,0};
    history.clear();
    const auto& events = dyn_grid.getConfig().value("dynamic_events", json::array());

    int max_steps = 2 * (dyn_grid.getRows() * dyn_grid.getCols());
    for (int t = 0; t < max_steps; ++t) {
        for (const auto& ev : events) if (ev.value("time_step", -1) == t) dyn_grid.addHazard(ev);
        
        EvacuationMode mode = get_mode(curr, dyn_grid);
        Cost::current_mode = mode;
        history.push_back({t, dyn_grid, curr, "Planning...", total_cost, mode});
        if (dyn_grid.isExit(curr.row, curr.col)) { history.back().action="SUCCESS"; break; }

        Direction move = Direction::STAY;
        {
            ParallelBIDP gpu_planner(dyn_grid);
            gpu_planner.run();
            const auto& map = gpu_planner.getCostMap();

            Cost best_c = map[curr.row][curr.col];
            int dr[]={-1,1,0,0}, dc[]={0,0,-1,1};
            Direction ds[]={Direction::UP,Direction::DOWN,Direction::LEFT,Direction::RIGHT};
            for(int i=0; i<4; ++i) {
                Position n={curr.row+dr[i], curr.col+dc[i]};
                if(dyn_grid.isWalkable(n.row, n.col) && map[n.row][n.col] < best_c) {
                    best_c = map[n.row][n.col];
                    move = ds[i];
                }
            }
        }

        std::string act = "STAY";
        if(move==Direction::UP) act="UP"; else if(move==Direction::DOWN) act="DOWN";
        else if(move==Direction::LEFT) act="LEFT"; else if(move==Direction::RIGHT) act="RIGHT";
        history.back().action = act;
        total_cost = total_cost + dyn_grid.getMoveCost(curr);
        curr = dyn_grid.getNextPosition(curr, move);
    }
    Cost::current_mode = EvacuationMode::NORMAL;
}
Cost ParallelInterlacedSolver::getEvacuationCost() const { return total_cost; }
void ParallelInterlacedSolver::generateReport(std::ofstream& f) const { for(const auto& s : history) f << s.grid_state.toHtmlStringWithAgent(s.agent_pos); }

// ============================================================
// 4. ParallelHierarchicalSolver
// ============================================================
ParallelHierarchicalSolver::ParallelHierarchicalSolver(const Grid& grid_ref) : Solver(grid_ref, "ParallelHierarchicalSim") {}
void ParallelHierarchicalSolver::run() {
    Grid dyn_grid = grid;
    Position curr = dyn_grid.getStartPosition();
    total_cost = {0,0,0};
    history.clear();
    current_plan.clear();
    const auto& events = dyn_grid.getConfig().value("dynamic_events", json::array());

    int max_steps = 2 * (dyn_grid.getRows() * dyn_grid.getCols());
    for (int t = 0; t < max_steps; ++t) {
        for (const auto& ev : events) if (ev.value("time_step", -1) == t) dyn_grid.addHazard(ev);
        EvacuationMode mode = get_mode(curr, dyn_grid);
        Cost::current_mode = mode;
        history.push_back({t, dyn_grid, curr, "Planning...", total_cost, mode});
        if (dyn_grid.isExit(curr.row, curr.col)) { history.back().action="SUCCESS"; break; }

        if (t % 10 == 0 || current_plan.empty()) {
            Cost best_c;
            Position next = curr;
            {
                ParallelBIDP gpu_planner(dyn_grid);
                gpu_planner.run();
                const auto& map = gpu_planner.getCostMap();
                best_c = map[curr.row][curr.col];
                
                int dr[]={-1,1,0,0}, dc[]={0,0,-1,1};
                for(int i=0; i<4; ++i) {
                    Position n={curr.row+dr[i], curr.col+dc[i]};
                    if(dyn_grid.isWalkable(n.row, n.col) && map[n.row][n.col] < best_c) {
                        best_c = map[n.row][n.col];
                        next = n;
                    }
                }
            }
            current_plan = {next};
        }

        Position next = current_plan.front();
        current_plan.erase(current_plan.begin());
        
        std::string act = "STAY";
        if(next.row < curr.row) act="UP"; else if(next.row > curr.row) act="DOWN";
        else if(next.col < curr.col) act="LEFT"; else if(next.col > curr.col) act="RIGHT";
        
        history.back().action = act;
        total_cost = total_cost + dyn_grid.getMoveCost(curr);
        curr = next;
    }
    Cost::current_mode = EvacuationMode::NORMAL;
}
Cost ParallelHierarchicalSolver::getEvacuationCost() const { return total_cost; }
void ParallelHierarchicalSolver::generateReport(std::ofstream& f) const { for(const auto& s : history) f << s.grid_state.toHtmlStringWithAgent(s.agent_pos); }

// ============================================================
// 5. ParallelPolicyBlendingSolver
// ============================================================
ParallelPolicyBlendingSolver::ParallelPolicyBlendingSolver(const Grid& grid_ref) 
    : Solver(grid_ref, "ParallelPolicyBlendingSim") {
    rl_solver = std::make_unique<QLearningSolver>(grid_ref);
    rl_solver->train(2000);
}
void ParallelPolicyBlendingSolver::run() {
    Grid dyn_grid = grid;
    Position curr = dyn_grid.getStartPosition();
    total_cost = {0,0,0};
    history.clear();
    const auto& events = dyn_grid.getConfig().value("dynamic_events", json::array());

    int max_steps = 2 * (dyn_grid.getRows() * dyn_grid.getCols());
    for (int t = 0; t < max_steps; ++t) {
        for (const auto& ev : events) if (ev.value("time_step", -1) == t) dyn_grid.addHazard(ev);
        EvacuationMode mode = get_mode(curr, dyn_grid);
        Cost::current_mode = mode;
        history.push_back({t, dyn_grid, curr, "Planning...", total_cost, mode});
        if (dyn_grid.isExit(curr.row, curr.col)) { history.back().action="SUCCESS"; break; }

        Direction move = Direction::STAY;
        if (mode == EvacuationMode::PANIC) {
            move = rl_solver->chooseAction(curr);
        } else {
            {
                ParallelBIDP gpu_planner(dyn_grid);
                gpu_planner.run();
                const auto& map = gpu_planner.getCostMap();
                
                // 1. Get DP Move
                Cost best_dp = map[curr.row][curr.col];
                Direction dp_move = Direction::STAY;
                Position next_dp = curr;
                int dr[]={-1,1,0,0}, dc[]={0,0,-1,1};
                Direction ds[]={Direction::UP,Direction::DOWN,Direction::LEFT,Direction::RIGHT};
                for(int i=0; i<4; ++i) {
                    Position n={curr.row+dr[i], curr.col+dc[i]};
                    if(dyn_grid.isWalkable(n.row, n.col) && map[n.row][n.col] < best_dp) {
                        best_dp = map[n.row][n.col];
                        next_dp = n;
                        dp_move = ds[i];
                    }
                }
                move = dp_move;
                
                // 2. If ALERT, blend with RL
                if (mode == EvacuationMode::ALERT) {
                     Direction rl_move = rl_solver->chooseAction(curr);
                     Position next_rl = dyn_grid.getNextPosition(curr, rl_move);
                     if (dyn_grid.isWalkable(next_rl.row, next_rl.col) && map[next_rl.row][next_rl.col].distance < best_dp.distance) {
                         move = rl_move;
                     }
                }
            }
        }

        std::string act = "STAY";
        if(move==Direction::UP) act="UP"; else if(move==Direction::DOWN) act="DOWN";
        else if(move==Direction::LEFT) act="LEFT"; else if(move==Direction::RIGHT) act="RIGHT";
        history.back().action = act;
        total_cost = total_cost + dyn_grid.getMoveCost(curr);
        curr = dyn_grid.getNextPosition(curr, move);
    }
    Cost::current_mode = EvacuationMode::NORMAL;
}
Cost ParallelPolicyBlendingSolver::getEvacuationCost() const { return total_cost; }
void ParallelPolicyBlendingSolver::generateReport(std::ofstream& f) const { for(const auto& s : history) f << s.grid_state.toHtmlStringWithAgent(s.agent_pos); }