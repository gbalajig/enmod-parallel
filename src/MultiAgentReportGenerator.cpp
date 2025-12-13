#include "enmod/MultiAgentReportGenerator.h"
#include "enmod/HtmlReportGenerator.h" // For writeHtmlHeader and writeHtmlFooter
#include <sstream> // For std::stringstream

MultiAgentReportGenerator::MultiAgentReportGenerator(const std::string& filename) {
    report_file.open(filename);
    writeHtmlHeader(report_file, "Multi-Agent Simulation Replay");
    report_file << "<h1>Multi-Agent Evacuation Replay</h1>\n";
}

void MultiAgentReportGenerator::add_timestep(int timestep, const Grid& grid, const std::vector<Position>& agent_positions) {
    report_file << "<h2>Timestep: " << timestep << "</h2>\n";
    report_file << toHtmlStringWithMultipleAgents(grid, agent_positions);
}

void MultiAgentReportGenerator::finalize_report() {
    writeHtmlFooter(report_file);
    report_file.close();
}

std::string MultiAgentReportGenerator::toHtmlStringWithMultipleAgents(const Grid& grid, const std::vector<Position>& agent_positions) const {
    std::stringstream ss;
    ss << "<table class='grid-table'><tbody>";
    for(int r = 0; r < grid.getRows(); ++r){
        ss << "<tr>";
        for(int c = 0; c < grid.getCols(); ++c){
            std::string content = "";
            bool agent_found = false;
            for(size_t i = 0; i < agent_positions.size(); ++i) {
                if (r == agent_positions[i].row && c == agent_positions[i].col) {
                    content = "A" + std::to_string(i + 1);
                    agent_found = true;
                    break;
                }
            }

            std::string class_name;
            if (!agent_found) {
                CellType cell_type = grid.getCellType({r, c});
                switch (cell_type) {
                    case CellType::WALL: class_name = "wall"; content = "W"; break;
                    case CellType::START: class_name = "start"; content = "S"; break;
                    case CellType::EXIT: class_name = "exit"; content = "E"; break;
                    case CellType::SMOKE: class_name = "smoke"; content = "~"; break;
                    case CellType::FIRE: class_name = "fire"; content = "F"; break;
                    default: break;
                }
            } else {
                class_name = "path";
            }
            
            ss << "<td class='grid-cell " << class_name << "'>" << content << "</td>";
        }
        ss << "</tr>";
    }
    ss << "</tbody></table>";
    return ss.str();
}