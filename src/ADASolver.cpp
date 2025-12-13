#include "enmod/ADASolver.h"
#include "enmod/Logger.h"
#include <queue>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>

// A* Node for the priority queue
struct ADAStarNode {
    Position pos;
    Cost cost;
    double f_score; // g_score + heuristic

    bool operator>(const ADAStarNode& other) const {
        return f_score > other.f_score;
    }
};

// Helper function to run A* algorithm with an inflated heuristic
std::vector<Position> run_anytime_astar(const Grid& grid, const Position& start_pos, double epsilon) {
    if (grid.getExitPositions().empty()) return {};

    Position goal_pos = grid.getExitPositions()[0]; // Simple goal selection

    std::priority_queue<ADAStarNode, std::vector<ADAStarNode>, std::greater<ADAStarNode>> open_set;
    std::map<Position, Position> came_from;
    std::map<Position, Cost> g_score;

    for(int r = 0; r < grid.getRows(); ++r) {
        for (int c = 0; c < grid.getCols(); ++c) {
            g_score[{r,c}] = {}; // Initialize with max cost
        }
    }

    g_score[start_pos] = {0, 0, 0};

    auto heuristic = [&](const Position& p) {
        return static_cast<double>(std::abs(p.row - goal_pos.row) + std::abs(p.col - goal_pos.col));
    };

    open_set.push({start_pos, {0, 0, 0}, epsilon * heuristic(start_pos)});

    while (!open_set.empty()) {
        Position current = open_set.top().pos;
        open_set.pop();

        if (current == goal_pos) {
            std::vector<Position> path;
            Position temp = current;
            while (came_from.count(temp)) {
                path.push_back(temp);
                temp = came_from[temp];
            }
            path.push_back(start_pos);
            std::reverse(path.begin(), path.end());
            return path;
        }

        int dr[] = {-1, 1, 0, 0};
        int dc[] = {0, 0, -1, 1};

        for (int i = 0; i < 4; ++i) {
            Position neighbor = {current.row + dr[i], current.col + dc[i]};

            if (grid.isWalkable(neighbor.row, neighbor.col)) {
                Cost tentative_g_score = g_score[current] + grid.getMoveCost(neighbor);

                if (tentative_g_score < g_score[neighbor]) {
                    came_from[neighbor] = current;
                    g_score[neighbor] = tentative_g_score;
                    double f_score = tentative_g_score.time + epsilon * heuristic(neighbor);
                    open_set.push({neighbor, tentative_g_score, f_score});
                }
            }
        }
    }
    return {}; // No path found
}

// THE FIX: Correct constructor initialization
ADASolver::ADASolver(const Grid& grid_ref)
    : Solver(grid_ref, "ADAStar"), epsilon(2.5) {}

void ADASolver::assessThreatAndSetMode(const Position& current_pos, const Grid& current_grid) {
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
}

void ADASolver::run() {
    // THE FIX: Use a local grid copy, consistent with other dynamic solvers
    Cost::current_mode = EvacuationMode::NORMAL;
    Grid dynamic_grid = grid;
    Position current_pos = dynamic_grid.getStartPosition();
    total_cost = {0, 0, 0};
    history.clear();

    const auto& events = dynamic_grid.getConfig().value("dynamic_events", json::array());

    for (int t = 0; t < 2 * (dynamic_grid.getRows() * dynamic_grid.getCols()); ++t) {
        for (const auto& event_cfg : events) {
            if (event_cfg.value("time_step", -1) == t) {
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

        auto path = run_anytime_astar(dynamic_grid, current_pos, epsilon);
        Position next_move = current_pos;
        std::string action = "STAY";

        if (path.size() > 1) {
            next_move = path[1];
            if(next_move.row < current_pos.row) action = "UP";
            else if(next_move.row > current_pos.row) action = "DOWN";
            else if(next_move.col < current_pos.col) action = "LEFT";
            else if(next_move.col > current_pos.col) action = "RIGHT";
        }

        if (path.empty()) {
             history.back().action = "FAILURE: No path found.";
             total_cost = {};
             break;
        }

        history.back().action = action;
        total_cost = total_cost + dynamic_grid.getMoveCost(current_pos);
        current_pos = next_move;
    }

    if (history.empty() || (history.back().action.find("SUCCESS") == std::string::npos && history.back().action.find("FAILURE") == std::string::npos)) {
         history.push_back({(int)history.size(), dynamic_grid, current_pos, "FAILURE: Timed out.", total_cost, current_mode});
         total_cost = {};
    }

    Cost::current_mode = EvacuationMode::NORMAL;
}

Cost ADASolver::getEvacuationCost() const {
    return total_cost;
}

void ADASolver::generateReport(std::ofstream& report_file) const {
    report_file << "<h2>Simulation History (ADA* Solver)</h2>\n";
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