#ifndef ENMOD_MULTI_AGENT_CPS_CONTROLLER_H
#define ENMOD_MULTI_AGENT_CPS_CONTROLLER_H

#include "Grid.h"
#include "HybridDPRLSolver.h"
#include "MultiAgentReportGenerator.h"
#include <string>
#include <vector>
#include <memory>

struct Agent {
    std::string id;
    Position position;
};

class MultiAgentCPSController {
public:
    MultiAgentCPSController(const json& initial_config, const std::string& report_path, int num_agents = 5);
    void run_simulation();

private:
    Grid master_grid;
    std::vector<Agent> agents;
    std::unique_ptr<HybridDPRLSolver> solver;
    MultiAgentReportGenerator report_generator;
    std::string report_path;

    // Added missing declarations
    void write_input_file(int timestep, const Agent& agent);
    Direction read_output_file(int timestep, const Agent& agent);
};

#endif // ENMOD_MULTI_AGENT_CPS_CONTROLLER_H