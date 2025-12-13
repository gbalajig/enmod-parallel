#include "enmod/PolicyGenerator.h"
#include "enmod/AVI.h"
#include <iomanip>
#include <iostream>
#include <set>
#include <fstream>

PolicyGenerator::PolicyGenerator(const Grid& grid_ref) 
    : Solver(grid_ref, "PolicyGen"), policy(grid_ref.getRows(), grid_ref.getCols()) {}

void PolicyGenerator::run() {
    AVI avi_solver(grid);
    avi_solver.run();
    const auto& cost_map = avi_solver.getCostMap();

    for (int r = 0; r < grid.getRows(); ++r) {
        for (int c = 0; c < grid.getCols(); ++c) {
            if (!grid.isWalkable(r, c) || grid.isExit(r, c)) {
                policy.setDirection({r,c}, Direction::STAY);
                continue;
            }

            Cost best_neighbor_cost;
            Direction best_direction = Direction::STAY;

            int dr[] = {-1, 1, 0, 0};
            int dc[] = {0, 0, -1, 1};
            Direction dirs[] = {Direction::UP, Direction::DOWN, Direction::LEFT, Direction::RIGHT};

            for (int i = 0; i < 4; ++i) {
                Position neighbor = {r + dr[i], c + dc[i]};
                if (grid.isValid(neighbor.row, neighbor.col)) {
                    if (cost_map[neighbor.row][neighbor.col] < best_neighbor_cost) {
                        best_neighbor_cost = cost_map[neighbor.row][neighbor.col];
                        best_direction = dirs[i];
                    }
                }
            }
            policy.setDirection({r, c}, best_direction);
        }
    }
    final_cost = calculatePolicyCost();
}

Cost PolicyGenerator::getEvacuationCost() const {
    return final_cost;
}

const Policy& PolicyGenerator::getPolicy() const {
    return policy;
}

Cost PolicyGenerator::calculatePolicyCost() const {
    Position current = grid.getStartPosition();
    Cost total_cost = {0, 0, 0};
    std::set<Position> visited;

    for (int i = 0; i < grid.getRows() * grid.getCols() + 1; ++i) {
        if (grid.isExit(current.row, current.col)) {
            return total_cost;
        }
        if (visited.count(current)) {
            return {}; 
        }
        visited.insert(current);
        
        Direction dir = policy.getDirection(current);
        total_cost = total_cost + grid.getMoveCost(current);
        current = grid.getNextPosition(current, dir);
    }
    return {};
}

void PolicyGenerator::generateReport(std::ofstream& report_file) const {
    report_file << "<h2>Final Policy Map (Generated from AVI Cost Map)</h2>\n";
    report_file << grid.toHtmlStringWithPolicy(policy);
}

