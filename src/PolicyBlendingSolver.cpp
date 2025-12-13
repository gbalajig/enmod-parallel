#include "enmod/PolicyBlendingSolver.h"
#include "enmod/BIDP.h"
#include "enmod/Logger.h"
#include <algorithm>
#include <cmath>

PolicyBlendingSolver::PolicyBlendingSolver(const Grid& grid_ref) 
    : Solver(grid_ref, "PolicyBlendingSim"), current_mode(EvacuationMode::NORMAL) {
    Cost::current_mode = EvacuationMode::NORMAL;
    rl_solver = std::make_unique<QLearningSolver>(grid_ref);
    rl_solver->train(5000);
}

void PolicyBlendingSolver::assessThreatAndSetMode(const Position& current_pos, const Grid& current_grid) {
    // This function is identical to the other dynamic solvers
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

void PolicyBlendingSolver::run() {
    Cost::current_mode = EvacuationMode::NORMAL;
    Grid dynamic_grid = grid;
    Position current_pos = dynamic_grid.getStartPosition();
    total_cost = {0, 0, 0};
    history.clear();

    const auto& events = dynamic_grid.getConfig().value("dynamic_events", json::array());

    for (int t = 0; t < 2 * (dynamic_grid.getRows() * dynamic_grid.getCols()); ++t) {
        for (const auto& event_cfg : events) {
            if (event_cfg.value("time_step", -1) == t) {
                dynamic_grid.addHazard(event_cfg);
            }
        }
        assessThreatAndSetMode(current_pos, dynamic_grid);
        Cost::current_mode = current_mode;
        history.push_back({t, dynamic_grid, current_pos, "Planning...", total_cost, current_mode});

        if (dynamic_grid.isExit(current_pos.row, current_pos.col)) {
            history.back().action = "SUCCESS: Reached Exit.";
            break;
        }

        Position next_move = current_pos;
        std::string action = "STAY";

        if (current_mode == EvacuationMode::PANIC) {
            Direction move_dir = rl_solver->chooseAction(current_pos);
            next_move = dynamic_grid.getNextPosition(current_pos, move_dir);
            if (move_dir == Direction::UP) action = "UP (RL)";
            else if (move_dir == Direction::DOWN) action = "DOWN (RL)";
            else if (move_dir == Direction::LEFT) action = "LEFT (RL)";
            else if (move_dir == Direction::RIGHT) action = "RIGHT (RL)";
        } else {
            BIDP step_planner(dynamic_grid);
            step_planner.run();
            const auto& cost_map = step_planner.getCostMap();
            
            Cost best_neighbor_cost_dp = cost_map[current_pos.row][current_pos.col];
            Position next_move_dp = current_pos;
            std::string action_dp = "STAY";
            
            int dr[] = {-1, 1, 0, 0};
            int dc[] = {0, 0, -1, 1};
            std::string actions[] = {"UP (DP)", "DOWN (DP)", "LEFT (DP)", "RIGHT (DP)"};

            for (int i = 0; i < 4; ++i) {
                Position neighbor = {current_pos.row + dr[i], current_pos.col + dc[i]};
                if (dynamic_grid.isWalkable(neighbor.row, neighbor.col)) {
                    if (cost_map[neighbor.row][neighbor.col] < best_neighbor_cost_dp) {
                        best_neighbor_cost_dp = cost_map[neighbor.row][neighbor.col];
                        next_move_dp = neighbor;
                        action_dp = actions[i];
                    }
                }
            }

            if (current_mode == EvacuationMode::NORMAL) {
                next_move = next_move_dp;
                action = action_dp;
            } else { // ALERT mode
                Direction move_dir_rl = rl_solver->chooseAction(current_pos);
                Position next_move_rl = dynamic_grid.getNextPosition(current_pos, move_dir_rl);
                
                if (dynamic_grid.isWalkable(next_move_rl.row, next_move_rl.col) &&
                    cost_map[next_move_rl.row][next_move_rl.col] < best_neighbor_cost_dp) {
                    next_move = next_move_rl;
                    if (move_dir_rl == Direction::UP) action = "UP (RL-Blend)";
                    else if (move_dir_rl == Direction::DOWN) action = "DOWN (RL-Blend)";
                    else if (move_dir_rl == Direction::LEFT) action = "LEFT (RL-Blend)";
                    else if (move_dir_rl == Direction::RIGHT) action = "RIGHT (RL-Blend)";
                } else {
                    next_move = next_move_dp;
                    action = action_dp;
                }
            }
        }
        
        // --- THE FIX: Validate the next position before moving ---
        if (!dynamic_grid.isWalkable(next_move.row, next_move.col)) {
            next_move = current_pos; // Stay put if the move is invalid
            action = "STAY (Invalid Move)";
        }
        // --- END FIX ---
        
        history.back().action = action;
        total_cost = total_cost + dynamic_grid.getMoveCost(current_pos);
        current_pos = next_move;
    }
     if(history.empty() || (history.back().action.find("SUCCESS") == std::string::npos && history.back().action.find("FAILURE") == std::string::npos)){
         history.push_back({(int)history.size(), dynamic_grid, current_pos, "FAILURE: Timed out.", total_cost, current_mode});
         total_cost = {};
     }
     Cost::current_mode = EvacuationMode::NORMAL;
}

Cost PolicyBlendingSolver::getEvacuationCost() const { return total_cost; }

void PolicyBlendingSolver::generateReport(std::ofstream& report_file) const {
    report_file << "<h2>Simulation History (Policy Blending Solver)</h2>\n";
    for (const auto& step : history) {
        std::string mode_str;
        switch(step.mode){
            case EvacuationMode::NORMAL: mode_str = "NORMAL"; break;
            case EvacuationMode::ALERT: mode_str = "ALERT"; break;
            case EvacuationMode::PANIC: mode_str = "PANIC"; break;
        }
        report_file << "<h3>Time Step: " << step.time_step << " (Mode: " << mode_str << ")</h3>\n";
        report_file << "<p><strong>Agent Position:</strong> (" << step.agent_pos.row << ", " << step.agent_pos.col << ")</p>\n";
        report_file << "<p><strong>Action Taken:</strong> " << step.action << "</p>\n";
        report_file << "<p><strong>Cumulative Cost:</strong> " << step.current_total_cost << "</p>\n";
        report_file << step.grid_state.toHtmlStringWithAgent(step.agent_pos);
    }
}