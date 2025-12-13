#include "enmod/SimulatedSensorNetwork.h"
#include <string>

SimulatedSensorNetwork::SimulatedSensorNetwork(const Grid& ground_truth_grid, const std::vector<Agent>* agents)
    : ground_truth(ground_truth_grid), active_agents(agents) {}

std::vector<SensorReading> SimulatedSensorNetwork::getAllReadings(double current_time) {
    std::vector<SensorReading> readings;

    // 1. Scan Grid for Hazards (Simulating fixed sensors in the building)
    for (int r = 0; r < ground_truth.getRows(); ++r) {
        for (int c = 0; c < ground_truth.getCols(); ++c) {
            Position pos = {r, c};
            CellType type = ground_truth.getCellType(pos);

            if (type == CellType::FIRE) {
                readings.push_back({
                    "THERMAL_" + std::to_string(r) + "_" + std::to_string(c),
                    SensorType::THERMAL_CAMERA,
                    pos,
                    500.0, // High temp
                    "FIRE_DETECTED",
                    current_time
                });
            }
            else if (type == CellType::SMOKE) {
                std::string intensity = ground_truth.getSmokeIntensity(pos);
                double density = (intensity == "heavy") ? 90.0 : 40.0;
                readings.push_back({
                    "SMOKE_" + std::to_string(r) + "_" + std::to_string(c),
                    SensorType::SMOKE_DETECTOR,
                    pos,
                    density,
                    intensity,
                    current_time
                });
            }
        }
    }

    // 2. Scan Agents (Simulating Wi-Fi/Bluetooth Beacons)
    if (active_agents) {
        for (const auto& agent : *active_agents) {
            readings.push_back({
                "BEACON_" + agent.id,
                SensorType::AGENT_BEACON,
                agent.position,
                1.0, // Signal strength
                agent.id,
                current_time
            });
        }
    }

    return readings;
}