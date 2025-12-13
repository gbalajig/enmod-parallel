#include "enmod/RLSolver.h"
#include "enmod/Logger.h"
#include <random>
#include <chrono>
#include <algorithm>

RLSolver::RLSolver(const Grid& grid_ref, const std::string& name)
    : Solver(grid_ref, name), policy(grid_ref.getRows(), grid_ref.getCols()) {}

const Policy& RLSolver::getPolicy() {
    generatePolicyFromValueTable();
    return policy;
}

void RLSolver::run() {
    train(5000); // Default training for static planners
    generatePolicyFromValueTable();
}

void RLSolver::train(int episodes) {
     for (int i = 0; i < episodes; ++i) {
        Position state = grid.getStartPosition();
        Direction action = chooseAction(state);

        for (int t = 0; t < grid.getRows() * grid.getCols(); ++t) {
            Position next_state = grid.getNextPosition(state, action);
            
            double reward = -1;
            if (!grid.isWalkable(next_state.row, next_state.col)) {
                reward = -100; next_state = state;
            } else if (grid.isExit(next_state.row, next_state.col)) {
                reward = 1000;
            } else if (grid.getCellType(next_state) == CellType::SMOKE) {
                reward = -20;
            } else if (grid.getCellType(next_state) == CellType::FIRE) {
                reward = -200;
            }
            
            Direction next_action = chooseAction(next_state);
            update(state, action, reward, next_state, next_action);

            state = next_state;
            action = next_action;

            if (grid.isExit(state.row, state.col)) {
                break;
            }
        }
    }
}

void RLSolver::generatePolicyFromValueTable() {
    for (int r = 0; r < grid.getRows(); ++r) {
        for (int c = 0; c < grid.getCols(); ++c) {
            Position pos = {r, c};
            if (grid.isWalkable(r, c)) {
                auto it = value_table.find(pos);
                if (it != value_table.end()) {
                    const auto& values = it->second;
                    int best_action_idx = static_cast<int>(std::distance(values.begin(), std::max_element(values.begin(), values.end())));
                    policy.setDirection(pos, static_cast<Direction>(best_action_idx));
                } else {
                    policy.setDirection(pos, Direction::NONE);
                }
            }
        }
    }
}