#include "enmod/DynamicFIDPSolver.h"
#include "enmod/FIDP.h"
#include "enmod/Logger.h"
#include <algorithm>
#include <cmath>

DynamicFIDPSolver::DynamicFIDPSolver(const Grid& grid_ref) 
    : Solver(grid_ref, "DynamicFIDPSim"), current_mode(EvacuationMode::NORMAL) {}

void DynamicFIDPSolver::assessThreatAndSetMode(const Position& current_pos, const Grid& current_grid) {
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

void DynamicFIDPSolver::run() {
    Cost::current_mode = EvacuationMode::NORMAL;
    Grid dynamic_grid = grid;
    Position current_pos = dynamic_grid.getStartPosition();
    total_cost = {0, 0, 0};
    history.clear();

    const auto& events = dynamic_grid.getConfig().value("dynamic_events", json::array());

    for (int t = 0; t < 2 * (dynamic_grid.getRows() * dynamic_grid.getCols()); ++t) {
        
        for (const auto& event_cfg : events) {
            if (event_cfg.value("time_step", -1) == t) {
                // THE FIX: Pass the entire event configuration object
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

        FIDP step_planner(dynamic_grid);
        step_planner.run(current_pos);
        
        auto path = step_planner.getEvacuationPath(current_pos);
        Position next_move = current_pos;
        std::string action = "STAY";

        if(path.size() > 1){
            next_move = path[1];
            if(next_move.row < current_pos.row) action = "UP";
            else if(next_move.row > current_pos.row) action = "DOWN";
            else if(next_move.col < current_pos.col) action = "LEFT";
            else if(next_move.col > current_pos.col) action = "RIGHT";
        }

        if(path.empty() || (path.size() == 1 && !(path[0] == current_pos))) {
             history.back().action = "FAILURE: No path found.";
             total_cost = {};
             break;
        }

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

Cost DynamicFIDPSolver::getEvacuationCost() const {
    return total_cost;
}

void DynamicFIDPSolver::generateReport(std::ofstream& report_file) const {
    report_file << "<h2>Simulation History (Turn-by-Turn using FIDP Planner)</h2>\n";
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

