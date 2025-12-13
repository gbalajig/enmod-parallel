#include "enmod/DQNSolver.h"
#include "enmod/Logger.h"
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>

// --- Helper Functions ---
static std::mt19937& get_rng() {
    static std::mt19937 rng(static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count()));
    return rng;
}

// --- NeuralNetwork Implementation ---
NeuralNetwork::NeuralNetwork(int in_size, int hid_size, int out_size)
    : input_size(in_size), hidden_size(hid_size), output_size(out_size) {
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    
    weights1.resize(hidden_size, std::vector<double>(input_size));
    bias1.resize(hidden_size);
    for (auto& row : weights1) for (auto& val : row) val = dist(get_rng());
    for (auto& val : bias1) val = dist(get_rng());

    weights2.resize(output_size, std::vector<double>(hidden_size));
    bias2.resize(output_size);
    for (auto& row : weights2) for (auto& val : row) val = dist(get_rng());
    for (auto& val : bias2) val = dist(get_rng());
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    std::vector<double> hidden(hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
        double sum = bias1[i];
        for (int j = 0; j < input_size; ++j) {
            sum += input[j] * weights1[i][j];
        }
        hidden[i] = relu(sum);
    }

    std::vector<double> output(output_size);
    for (int i = 0; i < output_size; ++i) {
        double sum = bias2[i];
        for (int j = 0; j < hidden_size; ++j) {
            sum += hidden[j] * weights2[i][j];
        }
        output[i] = sum; // No activation on final Q-values
    }
    return output;
}

void NeuralNetwork::train(const std::vector<double>& input, const std::vector<double>& target) {
    // This is a simplified backpropagation for a single training sample
    // A real implementation would use a more robust optimizer (like Adam) and batching.

    // Forward pass
    std::vector<double> hidden(hidden_size);
    std::vector<double> hidden_pre_activation(hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
        double sum = bias1[i];
        for (int j = 0; j < input_size; ++j) sum += input[j] * weights1[i][j];
        hidden_pre_activation[i] = sum;
        hidden[i] = relu(sum);
    }
    std::vector<double> output = predict(input);

    // Backward pass
    // Calculate output layer error
    std::vector<double> output_error(output_size);
    for (int i = 0; i < output_size; ++i) {
        output_error[i] = target[i] - output[i];
    }

    // Calculate hidden layer error
    std::vector<double> hidden_error(hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
        double error = 0.0;
        for (int j = 0; j < output_size; ++j) {
            error += output_error[j] * weights2[j][i];
        }
        hidden_error[i] = error * (hidden_pre_activation[i] > 0 ? 1 : 0); // Derivative of ReLU
    }

    // Update weights and biases
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            weights2[i][j] += learning_rate * output_error[i] * hidden[j];
        }
        bias2[i] += learning_rate * output_error[i];
    }
    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            weights1[i][j] += learning_rate * hidden_error[i] * input[j];
        }
        bias1[i] += learning_rate * hidden_error[i];
    }
}


// --- ReplayBuffer Implementation ---
ReplayBuffer::ReplayBuffer(size_t cap) : capacity(cap) {}

void ReplayBuffer::push(const Transition& transition) {
    if (memory.size() == capacity) memory.pop_front();
    memory.push_back(transition);
}

std::vector<Transition> ReplayBuffer::sample(size_t batch_size) {
    std::vector<Transition> batch;
    std::uniform_int_distribution<size_t> dist(0, memory.size() - 1);
    
    batch_size = std::min(batch_size, memory.size());
    for (size_t i = 0; i < batch_size; ++i) {
        batch.push_back(memory[dist(get_rng())]);
    }
    return batch;
}
size_t ReplayBuffer::size() const { return memory.size(); }


// --- DQNSolver Implementation ---
DQNSolver::DQNSolver(const Grid& grid_ref)
    : Solver(grid_ref, "DQN"),
      policy_net(5 * 5, 24, 4), // Input: 5x5 grid view, Hidden: 24, Output: 4 actions
      target_net(5 * 5, 24, 4),
      replay_buffer(10000)
{
    target_net = policy_net; // Initialize target network with policy network weights
    Logger::log(LogLevel::INFO, "DQN Solver initialized with a neural network.");
}

std::vector<double> DQNSolver::getStateRepresentation(const Grid& current_grid, const Position& pos) {
    // Create a 5x5 flattened vector representing the agent's local view
    std::vector<double> state_repr;
    state_repr.reserve(5 * 5);
    for (int r_offset = -2; r_offset <= 2; ++r_offset) {
        for (int c_offset = -2; c_offset <= 2; ++c_offset) {
            Position p = {pos.row + r_offset, pos.col + c_offset};
            if (!current_grid.isWalkable(p.row, p.col)) {
                state_repr.push_back(-1.0); // Wall
            } else if (current_grid.getCellType(p) == CellType::FIRE) {
                state_repr.push_back(1.0); // Fire
            } else if (current_grid.getCellType(p) == CellType::SMOKE) {
                state_repr.push_back(0.5); // Smoke
            } else {
                state_repr.push_back(0.0); // Empty
            }
        }
    }
    return state_repr;
}

Direction DQNSolver::chooseAction(const std::vector<double>& state_representation, const Grid& current_grid, const Position& pos) {
    std::uniform_real_distribution<> dist(0.0, 1.0);

    if (dist(get_rng()) < epsilon) {
        std::vector<Direction> valid_actions;
        if (current_grid.isWalkable(pos.row - 1, pos.col)) valid_actions.push_back(Direction::UP);
        if (current_grid.isWalkable(pos.row + 1, pos.col)) valid_actions.push_back(Direction::DOWN);
        if (current_grid.isWalkable(pos.row, pos.col - 1)) valid_actions.push_back(Direction::LEFT);
        if (current_grid.isWalkable(pos.row, pos.col + 1)) valid_actions.push_back(Direction::RIGHT);
        if (valid_actions.empty()) return Direction::STAY;
        std::uniform_int_distribution<size_t> action_dist(0, valid_actions.size() - 1);
        return valid_actions[action_dist(get_rng())];
    } else {
        auto q_values = policy_net.predict(state_representation);
        return static_cast<Direction>(std::distance(q_values.begin(), std::max_element(q_values.begin(), q_values.end())));
    }
}

void DQNSolver::replay() {
    if (replay_buffer.size() < batch_size) return;

    std::vector<Transition> batch = replay_buffer.sample(batch_size);
    
    for (const auto& transition : batch) {
        std::vector<double> q_values = policy_net.predict(transition.state);
        std::vector<double> q_values_next = target_net.predict(transition.next_state);
        double max_q_next = *std::max_element(q_values_next.begin(), q_values_next.end());
        
        double target_q = transition.reward;
        if (!transition.done) {
            target_q += gamma * max_q_next;
        }

        std::vector<double> target_q_values = q_values;
        target_q_values[static_cast<int>(transition.action)] = target_q;

        policy_net.train(transition.state, target_q_values);
    }
    
    if (epsilon > epsilon_min) epsilon *= epsilon_decay;

    if (++target_update_counter > 10) {
        target_net = policy_net;
        target_update_counter = 0;
    }
}

void DQNSolver::assessThreatAndSetMode(const Position& current_pos, const Grid& current_grid) {
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

void DQNSolver::run() {
    Cost::current_mode = EvacuationMode::NORMAL;
    Grid dynamic_grid = grid;
    Position current_pos = dynamic_grid.getStartPosition();
    total_cost = {0, 0, 0};
    history.clear();

    const auto& events = dynamic_grid.getConfig().value("dynamic_events", json::array());

    for (int t = 0; t < 2 * (dynamic_grid.getRows() * dynamic_grid.getCols()); ++t) {
        for (const auto& event_cfg : events) {
            if (event_cfg.value("time_step", -1) == t) dynamic_grid.addHazard(event_cfg);
        }

        assessThreatAndSetMode(current_pos, dynamic_grid);
        Cost::current_mode = current_mode;
        
        std::vector<double> state_repr = getStateRepresentation(dynamic_grid, current_pos);
        Direction move_dir = chooseAction(state_repr, dynamic_grid, current_pos);
        Position next_pos = dynamic_grid.getNextPosition(current_pos, move_dir);

        double reward = -1.0;
        bool done = false;
        if (!dynamic_grid.isWalkable(next_pos.row, next_pos.col)) {
            reward = -100.0;
            next_pos = current_pos;
        } else if (dynamic_grid.isExit(next_pos.row, next_pos.col)) {
            reward = 1000.0;
            done = true;
        } else if (dynamic_grid.getCellType(next_pos) == CellType::FIRE) {
            reward = -200.0;
        }

        std::vector<double> next_state_repr = getStateRepresentation(dynamic_grid, next_pos);
        replay_buffer.push({state_repr, move_dir, reward, next_state_repr, done});

        std::string action_str = "STAY";
        if (move_dir == Direction::UP) action_str = "UP";
        else if (move_dir == Direction::DOWN) action_str = "DOWN";
        else if (move_dir == Direction::LEFT) action_str = "LEFT";
        else if (move_dir == Direction::RIGHT) action_str = "RIGHT";
        history.push_back({t, dynamic_grid, current_pos, action_str, total_cost, current_mode});

        total_cost = total_cost + dynamic_grid.getMoveCost(current_pos);
        current_pos = next_pos;

        replay();
        
        if (done) {
            history.push_back({t + 1, dynamic_grid, current_pos, "SUCCESS: Reached Exit.", total_cost, current_mode});
            break;
        }
    }

    if(history.empty() || history.back().action.find("SUCCESS") == std::string::npos) {
         history.push_back({(int)history.size(), dynamic_grid, current_pos, "FAILURE: Timed out.", total_cost, current_mode});
         total_cost = {};
    }
}

Cost DQNSolver::getEvacuationCost() const { return total_cost; }

void DQNSolver::generateReport(std::ofstream& report_file) const {
    report_file << "<h2>Simulation History (DQN Solver)</h2>\n";
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