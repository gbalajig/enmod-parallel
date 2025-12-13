#include "enmod/CPSController.h"
#include "enmod/Logger.h"
#include <fstream>
#include <iostream>

CPSController::CPSController(const json& initial_config) : master_grid(initial_config) {
    agent_position = master_grid.getStartPosition();
    solver = std::make_unique<HybridDPRLSolver>(master_grid);
}

void CPSController::write_input_file(int timestep) {
    json input_data;
    input_data["agent_id"] = agent_id;
    input_data["timestep"] = timestep;
    input_data["current_position"] = {
        {"row", agent_position.row},
        {"col", agent_position.col}
    };
    
    // In a real system, this would come from sensors. Here we get it from the master grid.
    input_data["environment_update"] = master_grid.getConfig();

    std::ofstream o("agent_input.json");
    o << std::setw(4) << input_data << std::endl;
}

Direction CPSController::read_output_file() {
    std::ifstream i("agent_output.json");
    json output_data;
    i >> output_data;

    std::string move = output_data.at("next_move");
    if (move == "UP") return Direction::UP;
    if (move == "DOWN") return Direction::DOWN;
    if (move == "LEFT") return Direction::LEFT;
    if (move == "RIGHT") return Direction::RIGHT;
    return Direction::STAY;
}


void CPSController::run_simulation() {
    std::cout << "\n===== Starting Real-Time CPS Simulation =====\n";
    
    for (int t = 0; t < 2 * (master_grid.getRows() * master_grid.getCols()); ++t) {
        // 1. Simulate agent sending data to server
        write_input_file(t);
        Logger::log(LogLevel::INFO, "Timestep " + std::to_string(t) + ": Wrote agent_input.json");

        // 2. Server processes the data
        std::ifstream i("agent_input.json");
        json received_data;
        i >> received_data;

        Grid current_grid_state(received_data.at("environment_update"));
        Position current_agent_pos = {
            received_data.at("current_position").at("row"),
            received_data.at("current_position").at("col")
        };
        
        // 3. Server runs EnMod-DP and decides next move
        Direction next_move = solver->getNextMove(current_agent_pos, current_grid_state);

        // 4. Server sends command back to agent
        json output_data;
        output_data["agent_id"] = agent_id;
        std::string move_str = "STAY";
        if (next_move == Direction::UP) move_str = "UP";
        else if (next_move == Direction::DOWN) move_str = "DOWN";
        else if (next_move == Direction::LEFT) move_str = "LEFT";
        else if (next_move == Direction::RIGHT) move_str = "RIGHT";
        output_data["next_move"] = move_str;

        std::ofstream o("agent_output.json");
        o << std::setw(4) << output_data << std::endl;
        Logger::log(LogLevel::INFO, "Timestep " + std::to_string(t) + ": Wrote agent_output.json with move: " + move_str);

        // 5. Simulate agent receiving and executing the move
        Direction received_move = read_output_file();
        agent_position = master_grid.getNextPosition(agent_position, received_move);
        
        std::cout << "Timestep " << t << ": Agent at (" << agent_position.row << ", " << agent_position.col << ") received command: " << move_str << std::endl;

        // Update the master grid with any dynamic events for the next timestep
        for (const auto& event_cfg : master_grid.getConfig().value("dynamic_events", json::array())) {
            if (event_cfg.value("time_step", -1) == t + 1) {
                master_grid.addHazard(event_cfg);
            }
        }
        
        if (master_grid.isExit(agent_position.row, agent_position.col)) {
            std::cout << "SUCCESS: Agent reached the exit at timestep " << t << std::endl;
            Logger::log(LogLevel::INFO, "SUCCESS: Agent reached the exit.");
            break;
        }
    }
}