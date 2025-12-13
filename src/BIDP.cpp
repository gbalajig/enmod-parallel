#include "enmod/BIDP.h"
#include "enmod/HtmlReportGenerator.h"
#include <queue>
#include <vector>
#include <algorithm>

BIDP::BIDP(const Grid& grid_ref) : Solver(grid_ref, "BIDP") {}

void BIDP::run() {
    cost_map.assign(grid.getRows(), std::vector<Cost>(grid.getCols()));

    std::priority_queue<std::pair<Cost, Position>, std::vector<std::pair<Cost, Position>>, std::greater<std::pair<Cost, Position>>> pq;

    for (const auto& exit_pos : grid.getExitPositions()) {
        cost_map[exit_pos.row][exit_pos.col] = {0, 0, 0};
        pq.push({{0, 0, 0}, exit_pos});
    }

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
                // THE FIX: In a backward search, the cost of a move is associated with the cell
                // you are coming FROM, which is the 'next_pos' in this context.
                Cost move_cost = grid.getMoveCost(next_pos); 
                Cost new_cost = current_cost + move_cost;

                if (new_cost < cost_map[next_pos.row][next_pos.col]) {
                    cost_map[next_pos.row][next_pos.col] = new_cost;
                    pq.push({new_cost, next_pos});
                }
            }
        }
    }
}

Cost BIDP::getEvacuationCost() const {
    auto start_pos = grid.getStartPosition();
    if (grid.isValid(start_pos.row, start_pos.col)) {
        return cost_map[start_pos.row][start_pos.col];
    }
    return {};
}

void BIDP::generateReport(std::ofstream& report_file) const {
    report_file << "<h2>Final Cost Map (Cost to reach nearest Exit)</h2>\n";
    report_file << grid.toHtmlStringWithCost(cost_map);
}

const std::vector<std::vector<Cost>>& BIDP::getCostMap() const {
    return cost_map;
}

