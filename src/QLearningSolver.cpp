#include "enmod/QLearningSolver.h"
#include <random>
#include <chrono>
#include <algorithm>
#include <vector>

QLearningSolver::QLearningSolver(const Grid& grid_ref, const std::string& name) : RLSolver(grid_ref, name) {}

void QLearningSolver::run() {
    train(5000);
    generatePolicyFromValueTable();
}

Direction QLearningSolver::chooseAction(const Position& state) {
    static std::mt19937 rng(static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::uniform_real_distribution<> dist(0.0, 1.0);
    if (dist(rng) < epsilon) {
        return static_cast<Direction>(rng() % 4);
    } else {
        if (value_table.find(state) == value_table.end()) return static_cast<Direction>(rng() % 4);
        const auto& values = value_table.at(state);
        return static_cast<Direction>(std::distance(values.begin(), std::max_element(values.begin(), values.end())));
    }
}

void QLearningSolver::update(const Position& s, Direction a, double r, const Position& s_next, Direction /*a_next*/) {
    if (value_table.find(s) == value_table.end()) value_table[s] = {0.0, 0.0, 0.0, 0.0};
    if (value_table.find(s_next) == value_table.end()) value_table[s_next] = {0.0, 0.0, 0.0, 0.0};
    
    double old_value = value_table[s][static_cast<int>(a)];
    double next_max = *std::max_element(value_table[s_next].begin(), value_table[s_next].end());
    
    double new_value = (1 - alpha) * old_value + alpha * (r + gamma * next_max);
    value_table[s][static_cast<int>(a)] = new_value;
}

void QLearningSolver::train(int episodes) {
    RLSolver::train(episodes);
}

Cost QLearningSolver::getEvacuationCost() const {
    Position current = grid.getStartPosition();
    Cost total_cost = {0, 0, 0};
    std::vector<Position> path;
    for (int i = 0; i < grid.getRows() * grid.getCols() + 1; ++i) {
        path.push_back(current);
        if (grid.isExit(current.row, current.col)) return total_cost;
        Direction dir = policy.getDirection(current);
        if (dir == Direction::NONE) return {};
        total_cost = total_cost + grid.getMoveCost(current);
        current = grid.getNextPosition(current, dir);
        if(std::find(path.begin(), path.end(), current) != path.end()) return {};
    }
    return {};
}

void QLearningSolver::generateReport(std::ofstream& report_file) const {
    report_file << "<h2>Final Learned Policy (Q-Learning)</h2>\n";
    report_file << "<p>This policy was learned over 5000 episodes.</p>\n";
    report_file << grid.toHtmlStringWithPolicy(policy);
}