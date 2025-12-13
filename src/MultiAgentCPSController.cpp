#include "enmod/MultiAgentCPSController.h"
#include "enmod/Logger.h"
#include "enmod/SimulatedSensorNetwork.h"
#include "enmod/RealTimeSensorNetwork.h" // Include the RealTime sensor header
#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>

MultiAgentCPSController::MultiAgentCPSController(const json& initial_config, const std::string& report_path, int num_agents)
    : master_grid(initial_config),
      report_generator(report_path + "/multi_agent_report.html"),
      report_path(report_path) {

    std::filesystem::create_directory(report_path + "/agent_io");

    Position start_pos = master_grid.getStartPosition();
    for (int i = 0; i < num_agents; ++i) {
        Position agent_pos = {start_pos.row + i, start_pos.col};
        while (!master_grid.isWalkable(agent_pos.row, agent_pos.col)) {
            agent_pos.col++; 
            if (!master_grid.isValid(agent_pos.row, agent_pos.col)) {
                agent_pos.col--; 
                agent_pos.row++; 
            }
        }
        if (!master_grid.isValid(agent_pos.row, agent_pos.col)) {
            Logger::log(LogLevel::ERROR, "Could not place agent " + std::to_string(i) + " within grid bounds.");
            continue;
        }
        agents.push_back({"agent_" + std::to_string(i), agent_pos});
    }
    solver = std::make_unique<HybridDPRLSolver>(master_grid);
}

void MultiAgentCPSController::write_input_file(int timestep, const Agent& agent) {
    json input_data;
    input_data["agent_id"] = agent.id;
    input_data["timestep"] = timestep;
    input_data["current_position"] = {{"row", agent.position.row}, {"col", agent.position.col}};
    input_data["environment_update"] = master_grid.getConfig();
    std::ofstream o(report_path + "/agent_io/" + agent.id + "_input_t" + std::to_string(timestep) + ".json");
    o << std::setw(4) << input_data << std::endl;
}

Direction MultiAgentCPSController::read_output_file(int timestep, const Agent& agent) {
    std::ifstream i(report_path + "/agent_io/" + agent.id + "_output_t" + std::to_string(timestep) + ".json");
    json output_data;
    i >> output_data;
    std::string move = output_data.at("next_move");
    if (move == "UP") return Direction::UP;
    if (move == "DOWN") return Direction::DOWN;
    if (move == "LEFT") return Direction::LEFT;
    if (move == "RIGHT") return Direction::RIGHT;
    return Direction::STAY;
}
/*
void MultiAgentCPSController::run_simulation() {
    std::cout << "\n===== Starting Real-Time Multi-Agent CPS Simulation for " << master_grid.getName() << " =====\n";

    for (int t = 0; t < 2 * (master_grid.getRows() * master_grid.getCols()); ++t) {
        std::cout << "Timestep " << t << std::endl;

        for (const auto& event_cfg : master_grid.getConfig().value("dynamic_events", json::array())) {
            if (event_cfg.value("time_step", -1) == t) {
                master_grid.addHazard(event_cfg);
            }
        }

        std::vector<Position> agent_positions;
        for (const auto& agent : agents) {
            agent_positions.push_back(agent.position);
        }
        report_generator.add_timestep(t, master_grid, agent_positions);

        bool all_exited = true;
        for (auto& agent : agents) {
            if (master_grid.isExit(agent.position.row, agent.position.col)) {
                continue;
            }
            all_exited = false;

            write_input_file(t, agent);

            Grid agent_grid = master_grid;
            for (const auto& other_agent : agents) {
                if (agent.id != other_agent.id) {
                    agent_grid.setCellUnwalkable(other_agent.position);
                }
            }
            Direction next_move_dir = solver->getNextMove(agent.position, agent_grid);

            // --- THE FIX: Validate the next position before moving ---
            Position next_move_pos = master_grid.getNextPosition(agent.position, next_move_dir);
            if (!master_grid.isWalkable(next_move_pos.row, next_move_pos.col)) {
                next_move_dir = Direction::STAY; // Stay put if the move is invalid
            }
            // --- END FIX ---

            json output_data;
            output_data["agent_id"] = agent.id;
            std::string move_str = "STAY";
            if (next_move_dir == Direction::UP) move_str = "UP";
            else if (next_move_dir == Direction::DOWN) move_str = "DOWN";
            else if (next_move_dir == Direction::LEFT) move_str = "LEFT";
            else if (next_move_dir == Direction::RIGHT) move_str = "RIGHT";
            output_data["next_move"] = move_str;

            std::ofstream o(report_path + "/agent_io/" + agent.id + "_output_t" + std::to_string(t) + ".json");
            o << std::setw(4) << output_data << std::endl;

            Direction received_move = read_output_file(t, agent);
            agent.position = master_grid.getNextPosition(agent.position, received_move);
        }

        if (all_exited) {
            std::cout << "SUCCESS: All agents reached the exit." << std::endl;
            Logger::log(LogLevel::INFO, "SUCCESS: All agents reached the exit.");
            break;
        }
    }
*/

void MultiAgentCPSController::run_simulation() {
    std::cout << "\n===== Starting Real-Time Multi-Agent CPS Simulation (Digital Twin Mode) =====\n";

    // --- CONFIGURATION TOGGLE ---
    // Set to 'true' to read from "data/live_sensors.json" (fed by Python script)
    // Set to 'false' to use the internal simulator
    bool use_real_data = true; 
    std::string live_data_file = "data/live_sensors.json";
    // ----------------------------

    // Polymorphic pointer to handle either Real or Simulated sensors
    std::unique_ptr<ISensorNetwork> sensor_network;

    if (use_real_data) {
        std::cout << "[INFO] Connecting to Real-Time Sensor Feed: " << live_data_file << std::endl;
        // Ensure the file exists to prevent initial read errors
        std::ofstream outfile(live_data_file, std::ios::app); 
        outfile.close();
        
        sensor_network = std::make_unique<RealTimeSensorNetwork>(live_data_file);
    } else {
        std::cout << "[INFO] Using Internal Simulator for Sensors." << std::endl;
        sensor_network = std::make_unique<SimulatedSensorNetwork>(master_grid, &agents);
    }

    for (int t = 0; t < 2 * (master_grid.getRows() * master_grid.getCols()); ++t) {
        std::cout << "Timestep " << t << std::endl;

        // A. EVOLVE GROUND TRUTH (The "Real World")
        for (const auto& event_cfg : master_grid.getConfig().value("dynamic_events", json::array())) {
            if (event_cfg.value("time_step", -1) == t) {
                master_grid.addHazard(event_cfg);
            }
        }

        // B. SENSE (Digital Twin gets data from sensors)
        // FIX: Use '->' because sensor_network is a pointer
        std::vector<SensorReading> readings = sensor_network->getAllReadings(t);

        // C. UPDATE DIGITAL TWIN
        Grid digital_twin_grid(master_grid.getConfig()); // Re-initialize from static config
        digital_twin_grid.updateFromSensors(readings);   // Apply live sensor data

        // D. REPORTING
        std::vector<Position> agent_positions;
        for (const auto& agent : agents) agent_positions.push_back(agent.position);
        report_generator.add_timestep(t, master_grid, agent_positions); 

        // E. PLANNING
        bool all_exited = true;
        for (auto& agent : agents) {
            if (master_grid.isExit(agent.position.row, agent.position.col)) continue;
            all_exited = false;

            write_input_file(t, agent);

            // Create agent view from Digital Twin
            Grid agent_view = digital_twin_grid;
            for (const auto& other_agent : agents) {
                if (agent.id != other_agent.id) {
                    agent_view.setCellUnwalkable(other_agent.position);
                }
            }
            
            // Solver uses Digital Twin
            Direction next_move = solver->getNextMove(agent.position, agent_view);

            json output_data;
            output_data["agent_id"] = agent.id;
            std::string move_str = "STAY";
            if (next_move == Direction::UP) move_str = "UP";
            else if (next_move == Direction::DOWN) move_str = "DOWN";
            else if (next_move == Direction::LEFT) move_str = "LEFT";
            else if (next_move == Direction::RIGHT) move_str = "RIGHT";
            output_data["next_move"] = move_str;

            std::ofstream o(report_path + "/agent_io/" + agent.id + "_output_t" + std::to_string(t) + ".json");
            o << std::setw(4) << output_data << std::endl;

            Direction received_move = read_output_file(t, agent);
            agent.position = master_grid.getNextPosition(agent.position, received_move);
        }

        if (all_exited) {
            std::cout << "SUCCESS: All agents reached the exit." << std::endl;
            Logger::log(LogLevel::INFO, "SUCCESS: All agents reached the exit.");
            break;
        }
    }

    report_generator.finalize_report();
    std::cout << "\nMulti-agent simulation complete. Report generated at " << report_path << "/multi_agent_report.html\n";
}