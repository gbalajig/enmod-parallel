#include "enmod/DynamicActorCriticSolver.h"
#include "enmod/Logger.h"

DynamicActorCriticSolver::DynamicActorCriticSolver(const Grid& grid_ref) 
    : ActorCriticSolver(grid_ref) {
    solver_name = "DynamicActorCriticSim";
}

void DynamicActorCriticSolver::run() {
    Cost::current_mode = EvacuationMode::NORMAL;
    Grid dynamic_grid = grid;
    Position current_pos = dynamic_grid.getStartPosition();
    total_cost = {0, 0, 0};
    history.clear();
    
    train(2000); // Initial training phase

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

        Direction move_dir = chooseAction(current_pos);
        Position next_pos = dynamic_grid.getNextPosition(current_pos, move_dir);
        
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

        update(current_pos, move_dir, reward, next_pos, chooseAction(next_pos));
        
        std::string action = "STAY";
        if (move_dir == Direction::UP) action = "UP";
        else if (move_dir == Direction::DOWN) action = "DOWN";
        else if (move_dir == Direction::LEFT) action = "LEFT";
        else if (move_dir == Direction::RIGHT) action = "RIGHT";
        history.back().action = action;
        
        total_cost = total_cost + dynamic_grid.getMoveCost(current_pos);
        current_pos = next_pos;
    }

    if(history.empty() || (history.back().action.find("SUCCESS") == std::string::npos)){
         history.push_back({(int)history.size(), dynamic_grid, current_pos, "FAILURE: Timed out.", total_cost, EvacuationMode::NORMAL});
         total_cost = {};
     }
}
Cost DynamicActorCriticSolver::getEvacuationCost() const { return total_cost; }
void DynamicActorCriticSolver::generateReport(std::ofstream& report_file) const {
    report_file << "<h2>Simulation History (Turn-by-Turn with Online Actor-Critic)</h2>\n";
    for (const auto& step : history) {
        report_file << "<h3>Time Step: " << step.time_step << "</h3>\n";
        report_file << "<p><strong>Agent Position:</strong> (" << step.agent_pos.row << ", " << step.agent_pos.col << ")</p>\n";
        report_file << "<p><strong>Action Taken:</strong> " << step.action << "</p>\n";
        report_file << "<p><strong>Cumulative Cost:</strong> " << step.current_total_cost << "</p>\n";
        report_file << step.grid_state.toHtmlStringWithAgent(step.agent_pos);
    }
}

