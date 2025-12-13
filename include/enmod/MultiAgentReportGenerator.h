#ifndef ENMOD_MULTI_AGENT_REPORT_GENERATOR_H
#define ENMOD_MULTI_AGENT_REPORT_GENERATOR_H

#include "Grid.h"
#include <string>
#include <vector>
#include <fstream>

// Forward declarations for the helper functions
void writeHtmlHeader(std::ofstream& file, const std::string& title);
void writeHtmlFooter(std::ofstream& file);

class MultiAgentReportGenerator {
public:
    MultiAgentReportGenerator(const std::string& filename);
    void add_timestep(int timestep, const Grid& grid, const std::vector<Position>& agent_positions);
    void finalize_report();

private:
    std::ofstream report_file;
    std::string toHtmlStringWithMultipleAgents(const Grid& grid, const std::vector<Position>& agent_positions) const;
};

#endif // ENMOD_MULTI_AGENT_REPORT_GENERATOR_H