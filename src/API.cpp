#include "enmod/API.h"
#include "enmod/Logger.h"
#include <vector>
#include <algorithm>
#include <fstream>

API::API(const Grid& grid_ref) : Solver(grid_ref, "API"), policy(grid_ref.getRows(), grid_ref.getCols()) {}

void API::run() {
    if (grid.getExitPositions().empty()) return;
    auto first_exit = grid.getExitPositions()[0];
    for (int r = 0; r < grid.getRows(); ++r) {
        for (int c = 0; c < grid.getCols(); ++c) {
            if (r < first_exit.row) policy.setDirection({r, c}, Direction::DOWN);
            else if (r > first_exit.row) policy.setDirection({r, c}, Direction::UP);
            else if (c < first_exit.col) policy.setDirection({r, c}, Direction::RIGHT);
            else if (c > first_exit.col) policy.setDirection({r, c}, Direction::LEFT);
            else policy.setDirection({r, c}, Direction::STAY);
        }
    }

    bool policy_stable = false;
    int iteration = 0;
    while (!policy_stable) {
        policy_stable = true;
        iteration++;
        if(iteration > 100){ 
             Logger::log(LogLevel::ERROR, "API safety break triggered.");
             break;
        }

        std::vector<std::vector<Cost>> value_map(grid.getRows(), std::vector<Cost>(grid.getCols()));
        for (const auto& exit_pos : grid.getExitPositions()) {
            value_map[exit_pos.row][exit_pos.col] = {0, 0, 0};
        }
        
        for(int i = 0; i < grid.getRows() * grid.getCols(); ++i){
             for (int r = 0; r < grid.getRows(); ++r) {
                for (int c = 0; c < grid.getCols(); ++c) {
                    if(grid.isWalkable(r,c) && !grid.isExit(r,c)){
                        Position current_pos = {r,c};
                        Direction dir = policy.getDirection(current_pos);
                        Position next_pos = grid.getNextPosition(current_pos, dir);
                        if(grid.isWalkable(next_pos.row, next_pos.col)){
                            value_map[r][c] = grid.getMoveCost(current_pos) + value_map[next_pos.row][next_pos.col];
                        }
                    }
                }
            }
        }

        for (int r = 0; r < grid.getRows(); ++r) {
            for (int c = 0; c < grid.getCols(); ++c) {
                 if(grid.isWalkable(r,c) && !grid.isExit(r,c)){
                    Position current_pos = {r,c};
                    Direction old_action = policy.getDirection(current_pos);
                    
                    Cost best_cost;
                    Direction best_direction = Direction::STAY;

                    Direction dirs[] = {Direction::UP, Direction::DOWN, Direction::LEFT, Direction::RIGHT, Direction::STAY};
                    for(auto dir : dirs){
                        Position neighbor = grid.getNextPosition(current_pos, dir);
                        if(grid.isWalkable(neighbor.row, neighbor.col)){
                            Cost action_cost = grid.getMoveCost(current_pos) + value_map[neighbor.row][neighbor.col];
                            if(action_cost < best_cost){
                                best_cost = action_cost;
                                best_direction = dir;
                            }
                        }
                    }
                    
                    if(old_action != best_direction){
                        policy.setDirection(current_pos, best_direction);
                        policy_stable = false;
                    }
                 }
            }
        }
    }
}

Cost API::getEvacuationCost() const {
    return calculatePolicyCost();
}

Cost API::calculatePolicyCost() const {
    Position current = grid.getStartPosition();
    Cost total_cost = {0, 0, 0};
    std::vector<Position> path;

    for (int i = 0; i < grid.getRows() * grid.getCols() + 1; ++i) {
        path.push_back(current);
        if (grid.isExit(current.row, current.col)) {
            return total_cost;
        }

        Direction dir = policy.getDirection(current);
        total_cost = total_cost + grid.getMoveCost(current);
        current = grid.getNextPosition(current, dir);

        for (const auto& p : path) {
            if (p == current) {
                Logger::log(LogLevel::ERROR, "Loop detected in API policy. Returning infinite cost.");
                return {}; 
            }
        }
    }
    Logger::log(LogLevel::ERROR, "API policy failed to find exit. Returning infinite cost.");
    return {}; 
}

void API::generateReport(std::ofstream& report_file) const {
    report_file << "<h2>Final Policy Map (Policy Iteration)</h2>\n";
    report_file << grid.toHtmlStringWithPolicy(policy);
}

// NEW: Implementation of the getter
const Policy& API::getPolicy() const {
    return policy;
}

