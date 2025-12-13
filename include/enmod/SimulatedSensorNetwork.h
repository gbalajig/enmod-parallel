#ifndef ENMOD_SIMULATED_SENSOR_NETWORK_H
#define ENMOD_SIMULATED_SENSOR_NETWORK_H

#include "ISensorNetwork.h"
#include "Grid.h"
#include "MultiAgentCPSController.h" // To access Agent structs if needed

class SimulatedSensorNetwork : public ISensorNetwork {
public:
    SimulatedSensorNetwork(const Grid& ground_truth_grid, const std::vector<Agent>* agents);

    // Fetches data from the ground truth grid and agents
    std::vector<SensorReading> getAllReadings(double current_time) override;

private:
    const Grid& ground_truth;
    const std::vector<Agent>* active_agents;
};

#endif // ENMOD_SIMULATED_SENSOR_NETWORK_H