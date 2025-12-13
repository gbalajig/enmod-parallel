#include "enmod/ParallelDynamicSolvers.h"
#include "enmod/ParallelStaticSolvers.h"
#include "enmod/ParallelBIDP.h"
#include "enmod/Logger.h"

// --- Helper Base ---
ParallelDynamicSolverHelper::ParallelDynamicSolverHelper(const Grid& grid_ref, const std::string& name)
    : Solver(grid_ref, name), current_mode(EvacuationMode::NORMAL) {}

void ParallelDynamicSolverHelper::assessThreatAndSetMode(const Position& current_pos, const Grid& current_grid) {
    const auto& events = current_grid.getConfig().value("dynamic_events", json::array());
    current_mode = EvacuationMode::NORMAL; 
    for (const auto& event : events) {
        if (event.value("type", "") == "fire") {
            Position fire_pos = {event.at("position").at("row"), event.at("position").at("col")};
            if (current_grid.getCellType(fire_pos) == CellType::FIRE) {
                int radius = 1;
                if(event.value("size", "small") == "medium") radius = 2;
                if(event.value("size", "small") == "large") radius = 3;
                int dist = std::abs(current_pos.row - fire_pos.row) + std::abs(current_pos.col - fire_pos.col);
                if (dist <= 1) { current_mode = EvacuationMode::PANIC; return; }
                if (dist <= radius) { current_mode = EvacuationMode::ALERT; }
            }
        }
    }
    int dr[] = {-1, 1, 0, 0};
    int dc[] = {0, 0, -1, 1};
    for(int i = 0; i < 4; ++i) {
        Position neighbor = {current_pos.row + dr[i], current_pos.col + dc[i]};
        if(current_grid.getSmokeIntensity(neighbor) == "heavy"){
             if (current_mode != EvacuationMode::PANIC) current_mode = EvacuationMode::ALERT;
        }
    }
}

void ParallelDynamicSolverHelper::run() {
    Grid dynamic_grid = grid;
    Position current_pos = dynamic_grid.getStartPosition();
    total_cost = {0, 0, 0};
    history.clear();
    const auto& events = dynamic_grid.getConfig().value("dynamic_events", json::array());

    for (int t = 0; t < 2 * (dynamic_grid.getRows() * dynamic_grid.getCols()); ++t) {
        for (const auto& event_cfg : events) {
            if (event_cfg.value("time_step", -1) == t) dynamic_grid.addHazard(event_cfg);
        }
        assessThreatAndSetMode(current_pos, dynamic_grid);
        Cost::current_mode = current_mode;
        history.push_back({t, dynamic_grid, current_pos, "Planning...", total_cost, current_mode});

        if (dynamic_grid.isExit(current_pos.row, current_pos.col)) {
            history.back().action = "SUCCESS: Reached Exit.";
            break;
        }

        // --- CALL GPU STEP ---
        const auto& cost_map = run_gpu_step(dynamic_grid);
        // ---------------------

        Cost best_neighbor_cost = cost_map[current_pos.row][current_pos.col];
        Position best_next_move = current_pos;
        std::string action = "STAY";
        
        int dr[] = {-1, 1, 0, 0};
        int dc[] = {0, 0, -1, 1};
        std::string actions[] = {"UP", "DOWN", "LEFT", "RIGHT"};

        for (int i = 0; i < 4; ++i) {
            Position neighbor = {current_pos.row + dr[i], current_pos.col + dc[i]};
            if (dynamic_grid.isWalkable(neighbor.row, neighbor.col)) {
                // Minimize cost for BIDP/AVI/FIDP (assuming FIDP cost map represents cost-to-go or appropriate metric)
                if (cost_map[neighbor.row][neighbor.col] < best_neighbor_cost) {
                    best_neighbor_cost = cost_map[neighbor.row][neighbor.col];
                    best_next_move = neighbor;
                    action = actions[i];
                }
            }
        }
        
        if(best_next_move == current_pos && cost_map[current_pos.row][current_pos.col].distance == 2147483647){
             history.back().action = "FAILURE: No path found.";
             total_cost = {};
             break;
        }

        history.back().action = action;
        total_cost = total_cost + dynamic_grid.getMoveCost(current_pos);
        current_pos = best_next_move;
    }
    
     if(history.empty() || (history.back().action.find("SUCCESS") == std::string::npos && history.back().action.find("FAILURE") == std::string::npos)){
         history.push_back({(int)history.size(), dynamic_grid, current_pos, "FAILURE: Timed out.", total_cost, current_mode});
         total_cost = {};
     }
    Cost::current_mode = EvacuationMode::NORMAL;
}

Cost ParallelDynamicSolverHelper::getEvacuationCost() const { return total_cost; }

void ParallelDynamicSolverHelper::generateReport(std::ofstream& report_file) const {
    report_file << "<h2>Simulation History (" << solver_name << ")</h2>\n";
    for (const auto& step : history) {
        report_file << "<h3>Time Step: " << step.time_step << "</h3>\n";
        report_file << step.grid_state.toHtmlStringWithAgent(step.agent_pos);
    }
}

// --- Concrete Implementations ---
// We use static buffers to avoid reallocation, but re-construct solver to upload new grid data

// Dynamic FIDP
namespace { std::vector<std::vector<Cost>> temp_fidp_map; }
ParallelDynamicFIDPSolver::ParallelDynamicFIDPSolver(const Grid& grid) : ParallelDynamicSolverHelper(grid, "ParallelDynamicFIDPSim") {}
const std::vector<std::vector<Cost>>& ParallelDynamicFIDPSolver::run_gpu_step(const Grid& step_grid) {
    // FIX: ParallelFIDP (Forward) produces Cost-From-Start, which breaks gradient descent.
    // We use ParallelBIDP (Backward) to generate Cost-To-Goal, enabling the agent to navigate.
    ParallelBIDP solver(step_grid);
    solver.run();
    temp_fidp_map = solver.getCostMap();
    return temp_fidp_map;
}

// Dynamic AVI
namespace { std::vector<std::vector<Cost>> temp_avi_map; }
ParallelDynamicAVISolver::ParallelDynamicAVISolver(const Grid& grid) : ParallelDynamicSolverHelper(grid, "ParallelDynamicAVISim") {}
const std::vector<std::vector<Cost>>& ParallelDynamicAVISolver::run_gpu_step(const Grid& step_grid) {
    ParallelAVI solver(step_grid);
    solver.run();
    temp_avi_map = solver.getCostMap();
    return temp_avi_map;
}