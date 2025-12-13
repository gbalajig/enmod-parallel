#include "enmod/DynamicSARSASolver.h"
#include "enmod/Logger.h"

DynamicSARSASolver::DynamicSARSASolver(const Grid& grid_ref) 
    : SARSASolver(grid_ref) {
    solver_name = "DynamicSARSASim";
}

void DynamicSARSASolver::run() {
    Cost::current_mode = EvacuationMode::NORMAL;
    Grid dynamic_grid = grid;
    Position current_pos = dynamic_grid.getStartPosition();
    total_cost = {0, 0, 0};
    history.clear();
    
    train(1000); // Initial offline training
    Direction action = chooseAction(current_pos);

    const auto& events = dynamic_grid.getConfig().value("dynamic_events", json::array());

    for (int t = 0; t < 2 * (dynamic_grid.getRows() * dynamic_grid.getCols()); ++t) {
        for (const auto& event_cfg : events) {
            if (event_cfg.value("time_step", -1) == t) {
                dynamic_grid.addHazard(event_cfg);
            }
        }
        
        history.push_back({t, dynamic_grid, current_pos, "...", total_cost, EvacuationMode::NORMAL});

        if (dynamic_grid.isExit(current_pos.row, current_pos.col)) {
            history.back().action = "SUCCESS: Reached Exit.";
            break;
        }

        Position next_pos = dynamic_grid.getNextPosition(current_pos, action);
        
        double reward = -1;
        if (!dynamic_grid.isWalkable(next_pos.row, next_pos.col)) {
            reward = -100; next_pos = current_pos;
        } else if (dynamic_grid.isExit(next_pos.row, next_pos.col)) {
            reward = 1000;
        } else if (dynamic_grid.getCellType(next_pos) == CellType::FIRE) {
            reward = -200;
        } else if (dynamic_grid.getCellType(next_pos) == CellType::SMOKE) {
            reward = -20;
        }

        Direction next_action = chooseAction(next_pos);
        update(current_pos, action, reward, next_pos, next_action);
        
        std::string action_str = "STAY";
        if (action == Direction::UP) action_str = "UP";
        else if (action == Direction::DOWN) action_str = "DOWN";
        else if (action == Direction::LEFT) action_str = "LEFT";
        else if (action == Direction::RIGHT) action_str = "RIGHT";
        history.back().action = action_str;
        
        total_cost = total_cost + dynamic_grid.getMoveCost(current_pos);
        current_pos = next_pos;
        action = next_action;
    }

    if(history.empty() || (history.back().action.find("SUCCESS") == std::string::npos)){
         history.push_back({(int)history.size(), dynamic_grid, current_pos, "FAILURE: Timed out.", total_cost, EvacuationMode::NORMAL});
         total_cost = {};
     }
}
Cost DynamicSARSASolver::getEvacuationCost() const { return total_cost; }
void DynamicSARSASolver::generateReport(std::ofstream& report_file) const {
    report_file << "<h2>Simulation History (Turn-by-Turn with Online SARSA)</h2>\n";
    for (const auto& step : history) {
        report_file << "<h3>Time Step: " << step.time_step << "</h3>\n";
        report_file << "<p><strong>Agent Position:</strong> (" << step.agent_pos.row << ", " << step.agent_pos.col << ")</p>\n";
        report_file << "<p><strong>Action Taken:</strong> " << step.action << "</p>\n";
        report_file << "<p><strong>Cumulative Cost:</strong> " << step.current_total_cost << "</p>\n";
        report_file << step.grid_state.toHtmlStringWithAgent(step.agent_pos);
    }
}

