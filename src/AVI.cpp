#include "enmod/AVI.h"
#include "enmod/Logger.h"
#include <vector>
#include <algorithm>
#include <fstream>

AVI::AVI(const Grid& grid_ref) : Solver(grid_ref, "AVI") {}

const std::vector<std::vector<Cost>>& AVI::getCostMap() const {
    return cost_map;
}

void AVI::run() {
    cost_map.assign(grid.getRows(), std::vector<Cost>(grid.getCols()));

    for (const auto& exit_pos : grid.getExitPositions()) {
        cost_map[exit_pos.row][exit_pos.col] = {0, 0, 0};
    }

    bool changed = true;
    int iteration = 0;
    int max_iterations = 2 * (grid.getRows() * grid.getCols());

    while (changed) {
        changed = false;
        iteration++;
        if (iteration > max_iterations) {
            Logger::log(LogLevel::ERROR, "AVI safety break triggered after " + std::to_string(max_iterations) + " iterations. Possible infinite loop.");
            break;
        }

        for (int r = 0; r < grid.getRows(); ++r) {
            for (int c = 0; c < grid.getCols(); ++c) {
                if (grid.isWalkable(r, c) && !grid.isExit(r, c)) {
                    Cost min_cost;
                    
                    int dr[] = {-1, 1, 0, 0};
                    int dc[] = {0, 0, -1, 1};

                    for (int i = 0; i < 4; ++i) {
                        Position neighbor = {r + dr[i], c + dc[i]};
                        if (grid.isValid(neighbor.row, neighbor.col)) {
                            Cost neighbor_cost = cost_map[neighbor.row][neighbor.col];
                            // THE FIX: The cost of an action (moving from r,c to neighbor)
                            // is the cost incurred for taking a step FROM {r,c}.
                            Cost move_cost = grid.getMoveCost({r,c});
                            min_cost = std::min(min_cost, neighbor_cost + move_cost);
                        }
                    }

                    if (min_cost < cost_map[r][c]) {
                        cost_map[r][c] = min_cost;
                        changed = true;
                    }
                }
            }
        }
    }
}

Cost AVI::getEvacuationCost() const {
    auto start_pos = grid.getStartPosition();
    if (grid.isValid(start_pos.row, start_pos.col)) {
        return cost_map[start_pos.row][start_pos.col];
    }
    return {};
}

void AVI::generateReport(std::ofstream& report_file) const {
    report_file << "<h2>Final Cost Map (Value Iteration)</h2>\n";
    report_file << grid.toHtmlStringWithCost(cost_map);
}

