#include "enmod/AStarSolver.h"
#include <queue>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>

// A* Node for the priority queue
struct AStarNode {
    Position pos;
    Cost cost;
    double heuristic;

    bool operator>(const AStarNode& other) const {
        return (cost.time + heuristic) > (other.cost.time + other.heuristic);
    }
};

AStarSolver::AStarSolver(const Grid& grid_ref) : Solver(grid_ref, "AStar") {}

void AStarSolver::run() {
    std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<AStarNode>> open_set;
    std::map<Position, Position> came_from;
    std::map<Position, Cost> g_score;

    Position start_pos = grid.getStartPosition();
    g_score[start_pos] = {0, 0, 0};

    auto heuristic = [&](const Position& p) {
        double h = std::numeric_limits<double>::max();
        for (const auto& exit_pos : grid.getExitPositions()) {
            h = std::min(h, (double)std::abs(p.row - exit_pos.row) + std::abs(p.col - exit_pos.col));
        }
        return h;
    };

    open_set.push({start_pos, {0, 0, 0}, heuristic(start_pos)});

    while (!open_set.empty()) {
        Position current = open_set.top().pos;
        open_set.pop();

        if (grid.isExit(current.row, current.col)) {
            // Reconstruct path
            Position temp = current;
            while (came_from.count(temp)) {
                path.push_back(temp);
                temp = came_from[temp];
            }
            path.push_back(start_pos);
            std::reverse(path.begin(), path.end());
            total_cost = g_score[current];
            return;
        }

        int dr[] = {-1, 1, 0, 0};
        int dc[] = {0, 0, -1, 1};

        for (int i = 0; i < 4; ++i) {
            Position neighbor = {current.row + dr[i], current.col + dc[i]};

            if (grid.isWalkable(neighbor.row, neighbor.col)) {
                Cost tentative_g_score = g_score[current] + grid.getMoveCost(neighbor);

                if (!g_score.count(neighbor) || tentative_g_score < g_score[neighbor]) {
                    came_from[neighbor] = current;
                    g_score[neighbor] = tentative_g_score;
                    open_set.push({neighbor, tentative_g_score, heuristic(neighbor)});
                }
            }
        }
    }
}

Cost AStarSolver::getEvacuationCost() const {
    return total_cost;
}

void AStarSolver::generateReport(std::ofstream& report_file) const {
    report_file << "<h2>A* Evacuation Path</h2>\n";
    report_file << grid.toHtmlStringWithPath(path);
}