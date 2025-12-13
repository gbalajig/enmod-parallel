#include "enmod/ActorCriticSolver.h"
#include <random>
#include <chrono>
#include <algorithm>
#include <vector>
#include <numeric>

ActorCriticSolver::ActorCriticSolver(const Grid& grid_ref) : RLSolver(grid_ref, "ActorCritic") {}

void ActorCriticSolver::run() {
    train(10000); // Actor-Critic can take longer to converge
    generatePolicyFromValueTable();
}

Direction ActorCriticSolver::chooseAction(const Position& state) {
    static std::mt19937 rng(static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count()));
    
    if (value_table.find(state) == value_table.end()) {
        value_table[state] = {1.0, 1.0, 1.0, 1.0}; // Equal probabilities initially
    }
    
    // Choose action based on the probabilities in the actor's table
    std::discrete_distribution<> dist(value_table.at(state).begin(), value_table.at(state).end());
    return static_cast<Direction>(dist(rng));
}

void ActorCriticSolver::update(const Position& s, Direction a, double r, const Position& s_next, Direction /*a_next*/) {
    double actor_alpha = 0.01; // Actor often needs a smaller learning rate

    // --- Critic Update ---
    double old_state_value = state_value_table.count(s) ? state_value_table[s] : 0.0;
    double next_state_value = state_value_table.count(s_next) ? state_value_table[s_next] : 0.0;
    
    // Calculate the TD Error
    double td_error = r + gamma * next_state_value - old_state_value;
    
    // Update the Critic's value for the current state
    state_value_table[s] = old_state_value + alpha * td_error;

    // --- Actor Update ---
    if (value_table.find(s) == value_table.end()) {
        value_table[s] = {1.0, 1.0, 1.0, 1.0};
    }
    // Update the probability of taking that action based on the Critic's feedback (TD Error)
    int action_idx = static_cast<int>(a);
    value_table[s][action_idx] += actor_alpha * td_error;
    // Ensure probabilities don't go below a small value
    if(value_table[s][action_idx] < 0.01) value_table[s][action_idx] = 0.01;
}

Cost ActorCriticSolver::getEvacuationCost() const {
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

void ActorCriticSolver::generateReport(std::ofstream& report_file) const {
    report_file << "<h2>Final Learned Policy (Actor-Critic)</h2>\n";
    report_file << "<p>This policy was learned over 10000 episodes.</p>\n";
    report_file << grid.toHtmlStringWithPolicy(policy);
}

