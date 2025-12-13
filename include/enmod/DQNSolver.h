#ifndef ENMOD_DQN_SOLVER_H
#define ENMOD_DQN_SOLVER_H

#include "DynamicSolver.h"
#include "Types.h"
#include <vector>
#include <map>
#include <deque>

// Represents a single experience transition
struct Transition {
    std::vector<double> state; // Use a vector to represent the state for the NN
    Direction action;
    double reward;
    std::vector<double> next_state;
    bool done;
};

// A simple replay buffer to store and sample transitions
class ReplayBuffer {
public:
    explicit ReplayBuffer(size_t capacity);
    void push(const Transition& transition);
    std::vector<Transition> sample(size_t batch_size);
    size_t size() const;

private:
    std::deque<Transition> memory;
    size_t capacity;
};

// A very simple feedforward neural network
struct NeuralNetwork {
    NeuralNetwork(int input_size, int hidden_size, int output_size);
    std::vector<double> predict(const std::vector<double>& input);
    void train(const std::vector<double>& input, const std::vector<double>& target);

private:
    int input_size, hidden_size, output_size;
    std::vector<std::vector<double>> weights1; // Input to Hidden
    std::vector<double> bias1;                 // Hidden layer bias
    std::vector<std::vector<double>> weights2; // Hidden to Output
    std::vector<double> bias2;                 // Output layer bias
    double learning_rate = 0.001;

    // ReLU activation function
    double relu(double x) { return std::max(0.0, x); }
};


class DQNSolver : public Solver {
public:
    DQNSolver(const Grid& grid_ref);
    void run() override;
    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;

private:
    // --- Core DQN Components ---
    NeuralNetwork policy_net;
    NeuralNetwork target_net;
    ReplayBuffer replay_buffer;
    int batch_size = 32;
    int target_update_counter = 0;
    
    double gamma = 0.99;    // Discount factor
    double epsilon = 1.0;   // Exploration rate
    double epsilon_min = 0.01;
    double epsilon_decay = 0.995;

    // --- Simulation members ---
    std::vector<StepReport> history;
    Cost total_cost;
    EvacuationMode current_mode;

    // --- Methods ---
    std::vector<double> getStateRepresentation(const Grid& current_grid, const Position& pos);
    Direction chooseAction(const std::vector<double>& state_representation, const Grid& current_grid, const Position& pos);
    void replay(); // The training step
    void assessThreatAndSetMode(const Position& current_pos, const Grid& current_grid);
};

#endif // ENMOD_DQN_SOLVER_H