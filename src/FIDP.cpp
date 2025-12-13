#include "enmod/FIDP.h"
#include "enmod/HtmlReportGenerator.h"
#include <queue>
#include <vector>
#include <algorithm> 

FIDP::FIDP(const Grid& grid_ref) : Solver(grid_ref, "FIDP") {}

// Original run method for the static solver
void FIDP::run() {
    run(grid.getStartPosition());
}

// NEW overloaded run method that accepts a starting position
void FIDP::run(const Position& start_pos) {
    cost_map.assign(grid.getRows(), std::vector<Cost>(grid.getCols()));
    parent_map.assign(grid.getRows(), std::vector<Position>(grid.getCols(), {-1, -1}));

    std::priority_queue<std::pair<Cost, Position>, std::vector<std::pair<Cost, Position>>, std::greater<std::pair<Cost, Position>>> pq;

    cost_map[start_pos.row][start_pos.col] = {0, 0, 0};
    pq.push({{0, 0, 0}, start_pos});
    parent_map[start_pos.row][start_pos.col] = start_pos;

    int dr[] = {-1, 1, 0, 0};
    int dc[] = {0, 0, -1, 1};

    while (!pq.empty()) {
        auto [current_cost, current_pos] = pq.top();
        pq.pop();

        if (cost_map[current_pos.row][current_pos.col] < current_cost) {
            continue;
        }

        for (int i = 0; i < 4; ++i) {
            Position next_pos = {current_pos.row + dr[i], current_pos.col + dc[i]};

            if (grid.isWalkable(next_pos.row, next_pos.col)) {
                Cost move_cost = grid.getMoveCost(next_pos);
                Cost new_cost = current_cost + move_cost;

                if (new_cost < cost_map[next_pos.row][next_pos.col]) {
                    cost_map[next_pos.row][next_pos.col] = new_cost;
                    pq.push({new_cost, next_pos});
                    parent_map[next_pos.row][next_pos.col] = current_pos;
                }
            }
        }
    }
}

// UPDATED to use a start position for tracing
std::vector<Position> FIDP::getEvacuationPath(const Position& start_pos) const {
    std::vector<Position> path;
    Cost best_cost;
    Position best_exit = {-1, -1};

    for (const auto& exit_pos : grid.getExitPositions()) {
        if (cost_map[exit_pos.row][exit_pos.col] < best_cost) {
            best_cost = cost_map[exit_pos.row][exit_pos.col];
            best_exit = exit_pos;
        }
    }

    if (best_exit.row == -1) {
        return path; 
    }

    Position current = best_exit;
    while (!(current == start_pos) && current.row != -1) {
        path.push_back(current);
        if (parent_map[current.row][current.col] == current) {
            path.clear(); // Invalid path
            break;
        }
        current = parent_map[current.row][current.col];
    }
    path.push_back(start_pos); // Add the start position
    
    std::reverse(path.begin(), path.end());
    return path;
}


Cost FIDP::getEvacuationCost() const {
    Cost best_cost;
    for (const auto& exit_pos : grid.getExitPositions()) {
        if (cost_map[exit_pos.row][exit_pos.col] < best_cost) {
            best_cost = cost_map[exit_pos.row][exit_pos.col];
        }
    }
    return best_cost;
}

void FIDP::generateReport(std::ofstream& report_file) const {
    report_file << "<h2>Final Cost Map (Cost from Start)</h2>\n";
    report_file << grid.toHtmlStringWithCost(cost_map);
    report_file << "<h2>Evacuation Path</h2>\n";
    auto path = getEvacuationPath(grid.getStartPosition());
    report_file << grid.toHtmlStringWithPath(path);
}

