// src/DStarLiteSolver.cpp

#include "enmod/DStarLiteSolver.h"
#include "enmod/Logger.h"
#include <algorithm>
#include <cmath>
#include <map>

DStarLiteSolver::DStarLiteSolver(const Grid& grid_ref)
    : Solver(grid_ref, "DStarLiteSim"), dynamic_grid(grid_ref) {}

double DStarLiteSolver::heuristic(const Position& a, const Position& b) {
    return std::abs(a.row - b.row) + std::abs(a.col - b.col);
}

Key DStarLiteSolver::calculateKey(const Position& p) {
    if (maze.find(p) == maze.end()) {
        maze[p] = DStarNode();
    }
    double min_val = std::min(maze[p].g, maze[p].rhs);
    return {min_val + heuristic(start_pos, p) + k_m, min_val};
}

void DStarLiteSolver::initialize() {
    maze.clear();
    while (!open_set.empty()) open_set.pop();
    k_m = 0.0;

    start_pos = grid.getStartPosition();
    if (!grid.getExitPositions().empty()) {
        goal_pos = grid.getExitPositions()[0];
    } else {
        goal_pos = {-1, -1};
    }

    maze[goal_pos].rhs = 0;
    open_set.push({calculateKey(goal_pos), goal_pos});
}

void DStarLiteSolver::updateVertex(const Position& p) {
    if (!(p == goal_pos)) {
        double min_rhs = std::numeric_limits<double>::infinity();
        int dr[] = {-1, 1, 0, 0};
        int dc[] = {0, 0, -1, 1};

        for (int i = 0; i < 4; ++i) {
            Position successor = {p.row + dr[i], p.col + dc[i]};
            if (dynamic_grid.isWalkable(successor.row, successor.col)) {
                if (maze.find(successor) == maze.end()) maze[successor] = DStarNode();
                Cost move_cost = dynamic_grid.getMoveCost(successor);
                double cost_val = (move_cost.smoke * 1000) + (move_cost.time * 10) + (move_cost.distance * 1);
                min_rhs = std::min(min_rhs, maze[successor].g + cost_val);
            }
        }
        maze[p].rhs = min_rhs;
    }

    // if p is in the open set, remove it
    // (this is not efficient, but it is correct)
    std::priority_queue<std::pair<Key, Position>, std::vector<std::pair<Key, Position>>, std::greater<std::pair<Key, Position>>> new_open_set;
    while (!open_set.empty())
    {
        if (open_set.top().second.row != p.row || open_set.top().second.col != p.col)
        {
            new_open_set.push(open_set.top());
        }
        open_set.pop();
    }
    open_set = new_open_set;

    if (maze[p].g != maze[p].rhs) {
        open_set.push({calculateKey(p), p});
    }
}


void DStarLiteSolver::computeShortestPath() {
    while (!open_set.empty() && (open_set.top().first < calculateKey(start_pos) || maze[start_pos].g != maze[start_pos].rhs)) {
        Key k_old = open_set.top().first;
        Position u = open_set.top().second;
        open_set.pop();

        Key k_new = calculateKey(u);
        if (k_old < k_new) {
            open_set.push({k_new, u});
            continue;
        }

        if (maze[u].g > maze[u].rhs) {
            maze[u].g = maze[u].rhs;
            int dr[] = {-1, 1, 0, 0};
            int dc[] = {0, 0, -1, 1};
            for (int i = 0; i < 4; ++i) {
                Position predecessor = {u.row + dr[i], u.col + dc[i]};
                if (dynamic_grid.isWalkable(predecessor.row, predecessor.col)) {
                    updateVertex(predecessor);
                }
            }
        } else {
            maze[u].g = std::numeric_limits<double>::infinity();
            updateVertex(u);
            int dr[] = {-1, 1, 0, 0};
            int dc[] = {0, 0, -1, 1};
            for (int i = 0; i < 4; ++i) {
                Position predecessor = {u.row + dr[i], u.col + dc[i]};
                 if (dynamic_grid.isWalkable(predecessor.row, predecessor.col)) {
                    updateVertex(predecessor);
                }
            }
        }
    }
}


void DStarLiteSolver::run() {
    Cost::current_mode = EvacuationMode::NORMAL;
    total_cost = {0, 0, 0};
    history.clear();

    initialize();
    if (goal_pos.row == -1) {
        history.push_back({0, dynamic_grid, start_pos, "FAILURE: No exit found.", total_cost, EvacuationMode::NORMAL});
        total_cost = {};
        return;
    }
    computeShortestPath();

    const auto& events = dynamic_grid.getConfig().value("dynamic_events", json::array());

    for (int t = 0; t < 2 * (grid.getRows() * grid.getCols()); ++t) {
        history.push_back({t, dynamic_grid, start_pos, "...", total_cost, EvacuationMode::NORMAL});

        if (start_pos == goal_pos) {
            history.back().action = "SUCCESS: Reached Exit.";
            break;
        }

        if (maze.find(start_pos) == maze.end() || maze[start_pos].g == std::numeric_limits<double>::infinity()) {
            history.back().action = "FAILURE: No path found.";
            total_cost = {};
            break;
        }

        Position next_move = start_pos;
        double min_cost = std::numeric_limits<double>::infinity();
        std::string action = "STAY";

        int dr[] = {-1, 1, 0, 0};
        int dc[] = {0, 0, -1, 1};
        std::string actions[] = {"UP", "DOWN", "LEFT", "RIGHT"};
        
        for (int i = 0; i < 4; ++i) {
            Position successor = {start_pos.row + dr[i], start_pos.col + dc[i]};
            if (dynamic_grid.isWalkable(successor.row, successor.col)) {
                if (maze.find(successor) == maze.end()) maze[successor] = DStarNode();
                Cost move_cost = dynamic_grid.getMoveCost(successor);
                double cost_val = (move_cost.smoke * 1000) + (move_cost.time * 10) + (move_cost.distance * 1);
                double cost = maze[successor].g + cost_val;
                if (cost < min_cost) {
                    min_cost = cost;
                    next_move = successor;
                    action = actions[i];
                }
            }
        }

        history.back().action = action;
        total_cost = total_cost + dynamic_grid.getMoveCost(start_pos);
        last_pos = start_pos;
        start_pos = next_move;

        bool cost_changed = false;
        std::vector<Position> changed_cells;
        for (const auto& event_cfg : events) {
            if (event_cfg.value("time_step", -1) == t + 1) {
                Position changed_pos = {event_cfg.at("position").at("row"), event_cfg.at("position").at("col")};
                dynamic_grid.addHazard(event_cfg);
                changed_cells.push_back(changed_pos);
                cost_changed = true;
            }
        }

        if (cost_changed) {
            k_m += heuristic(last_pos, start_pos);
            for (const auto& p : changed_cells) {
                updateVertex(p);
                 for (int i = 0; i < 4; ++i) {
                    Position neighbor = {p.row + dr[i], p.col + dc[i]};
                    if (dynamic_grid.isWalkable(neighbor.row, neighbor.col)) {
                        updateVertex(neighbor);
                    }
                }
            }
            computeShortestPath();
        }
    }

    if (history.empty() || (history.back().action.find("SUCCESS") == std::string::npos && history.back().action.find("FAILURE") == std::string::npos)) {
        history.push_back({(int)history.size(), dynamic_grid, start_pos, "FAILURE: Timed out.", total_cost, EvacuationMode::NORMAL});
        total_cost = {};
    }
}


Cost DStarLiteSolver::getEvacuationCost() const {
    return total_cost;
}

void DStarLiteSolver::generateReport(std::ofstream& report_file) const {
    report_file << "<h2>Simulation History (D* Lite Solver)</h2>\n";
    for (const auto& step : history) {
        report_file << "<h3>Time Step: " << step.time_step << "</h3>\n";
        report_file << "<p><strong>Agent Position:</strong> (" << step.agent_pos.row << ", " << step.agent_pos.col << ")</p>\n";
        report_file << "<p><strong>Action Taken:</strong> " << step.action << "</p>\n";
        report_file << "<p><strong>Cumulative Cost:</strong> " << step.current_total_cost << "</p>\n";
        report_file << step.grid_state.toHtmlStringWithAgent(step.agent_pos);
    }
}