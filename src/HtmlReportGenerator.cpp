#include "enmod/HtmlReportGenerator.h"
#include <fstream>
#include <map>
#include <algorithm>
#include <vector>

void writeHtmlHeader(std::ofstream& file, const std::string& title) {
    file << "<!DOCTYPE html>\n<html lang='en'>\n<head>\n";
    file << "<meta charset='UTF-8'>\n";
    file << "<meta name='viewport' content='width=device-width, initial-scale=1.0'>\n";
    file << "<title>" << title << "</title>\n";
    file << "<style>\n";
    file << "body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f4f4f9; color: #333; }\n";
    file << "h1, h2 { color: #444; border-bottom: 2px solid #ddd; padding-bottom: 10px; }\n";
    file << "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }\n";
    file << "th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }\n";
    file << "th { background-color: #007bff; color: white; }\n";
    file << "td.row-header { background-color: #f2f2f2; font-weight: bold; text-align: left; }\n";
    file << ".container { max-width: 1400px; margin: auto; background: white; padding: 20px; border-radius: 8px; }\n";
    file << ".grid-table { border-spacing: 0; border-collapse: separate; }\n";
    file << ".grid-cell { width: 25px; height: 25px; text-align: center; vertical-align: middle; font-size: 12px; border: 1px solid #ccc; }\n";
    file << ".wall { background-color: #343a40; color: white; }\n";
    file << ".start { background-color: #28a745; color: white; }\n";
    file << ".exit { background-color: #dc3545; color: white; }\n";
    file << ".smoke { background-color: #6c757d; color: white; }\n";
    file << ".fire { background-color: #ffc107; color: black; animation: pulse 1s infinite; }\n";
    file << ".path { background-color: #17a2b8; }\n";
    file << "@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.6; } 100% { opacity: 1; } }\n";
    file << "</style>\n</head>\n<body>\n<div class='container'>\n";
}

void writeHtmlFooter(std::ofstream& file) {
    file << "</div>\n</body>\n</html>";
}

void HtmlReportGenerator::generateInitialGridReport(const Grid& grid, const std::string& path) {
    std::string file_path = path + "/_Initial_Grid.html";
    std::ofstream report_file(file_path);
    if (!report_file) return;
    writeHtmlHeader(report_file, "Initial Grid - " + grid.getName());
    report_file << "<h1>Initial Grid: " << grid.getName() << " (" << grid.getRows() << "x" << grid.getCols() << ")</h1>\n";
    report_file << grid.toHtmlString();
    writeHtmlFooter(report_file);
}

void HtmlReportGenerator::generateSolverReport(const Solver& solver, const std::string& path) {
    std::string file_path = path + "/" + solver.getName() + "_Report.html";
    std::ofstream report_file(file_path);
    if (!report_file) return;
    writeHtmlHeader(report_file, solver.getName() + " Report");
    report_file << "<h1>" << solver.getName() << " Report</h1>\n";
    solver.generateReport(report_file);
    writeHtmlFooter(report_file);
}

void HtmlReportGenerator::generateSummaryReport(const std::vector<Result>& results, const std::string& path) {
    std::string file_path = path + "/_Summary_Report.html";
    std::ofstream report_file(file_path);
    if (!report_file) return;

    writeHtmlHeader(report_file, "Simulation Summary Report");
    report_file << "<h1>Final Simulation Summary</h1>\n";

    std::vector<std::string> scenarios;
    std::map<std::string, bool> seen_scenarios;
    for (const auto& res : results) {
        if (!seen_scenarios[res.scenario_name]) {
            scenarios.push_back(res.scenario_name);
            seen_scenarios[res.scenario_name] = true;
        }
    }
     std::sort(scenarios.begin(), scenarios.end(), [](const std::string& a, const std::string& b){
        return std::stoi(a.substr(0, a.find('x'))) < std::stoi(b.substr(0, b.find('x')));
    });

    std::vector<std::string> static_dp_solvers = {"BIDP", "FIDP", "API"};
    std::vector<std::string> static_heuristic_solvers = {"AStar"};
    std::vector<std::string> static_rl_solvers = {"QLearning", "SARSA", "ActorCritic"};
    std::vector<std::string> dynamic_dp_solvers = {"DynamicBIDPSim", "DynamicFIDPSim", "DynamicAVISim", "DynamicAPISim"};
    std::vector<std::string> dynamic_heuristic_solvers = {"DynamicAStarSim"}; 
    std::vector<std::string> dynamic_rl_solvers = {"DynamicQLearningSim", "DynamicSARSASim", "DynamicActorCriticSim"};
    std::vector<std::string> advanced_heuristic_solvers = {"DynamicHPAStar", "ADASolver", "DStarLiteSim"};
    std::vector<std::string> deep_rl_solvers = {"DQN"};
    std::vector<std::string> hybrid_solvers = {"HybridDPRLSim", "AdaptiveCostSim", "InterlacedSim", "HierarchicalSim", "PolicyBlendingSim", "RLEnhancedAStar"};
    std::vector<std::string> cuda_solvers = {"ParallelBIDP", "ParallelFIDP", "ParallelAVI","ParallelDynamicBIDPSim", "ParallelDynamicFIDPSim", "ParallelDynamicAVISim","ParallelHybridDPRLSim", "ParallelAdaptiveCostSim", "ParallelInterlacedSim", "ParallelHierarchicalSim", "ParallelPolicyBlendingSim" };


    report_file << "<table>\n";
    
    report_file << "<thead><tr><th rowspan='2'>Algorithm</th>";
    for(const auto& scn : scenarios){
        report_file << "<th colspan='5'>" << scn << "</th>";
    }
    report_file << "</tr>\n";

    report_file << "<tr>";
    for(size_t i = 0; i < scenarios.size(); ++i){
        report_file << "<th>Smoke</th><th>Time</th><th>Dist</th><th>W. Cost</th><th>Exec. Time (ms)</th>";
    }
    report_file << "</tr></thead>\n<tbody>";

    auto write_solver_rows = [&](const std::vector<std::string>& solver_names){
        for(const auto& solver_name : solver_names){
            report_file << "<tr><td>" << solver_name << "</td>";
            for(const auto& scn : scenarios){
                auto it = std::find_if(results.begin(), results.end(), [&](const Result& r){
                    return r.scenario_name == scn && r.solver_name == solver_name;
                });
                if(it != results.end()){
                    if (it->cost.distance == MAX_COST) {
                        report_file << "<td colspan='5'>FAILURE / N/A</td>";
                    } else {
                        report_file << "<td>" << it->cost.smoke << "</td>";
                        report_file << "<td>" << it->cost.time << "</td>";
                        report_file << "<td>" << it->cost.distance << "</td>";
                        report_file << "<td>" << static_cast<int>(it->weighted_cost) << "</td>";
                        report_file << "<td>" << it->execution_time << "</td>";
                    }
                } else {
                     report_file << "<td colspan='5'>Not Run</td>";
                }
            }
            report_file << "</tr>\n";
        }
    };

    report_file << "<tr><td colspan='" << (1 + scenarios.size() * 5) << "' class='row-header'>Static Planners (Model-Based DP)</td></tr>";
    write_solver_rows(static_dp_solvers);

    report_file << "<tr><td colspan='" << (1 + scenarios.size() * 5) << "' class='row-header'>Static Planners (Heuristic)</td></tr>";
    write_solver_rows(static_heuristic_solvers);
    
    report_file << "<tr><td colspan='" << (1 + scenarios.size() * 5) << "' class='row-header'>Static Planners (Model-Free RL Training)</td></tr>";
    write_solver_rows(static_rl_solvers);

    report_file << "<tr><td colspan='" << (1 + scenarios.size() * 5) << "' class='row-header'>Dynamic Simulators (DP-Based)</td></tr>";
    write_solver_rows(dynamic_dp_solvers);

    report_file << "<tr><td colspan='" << (1 + scenarios.size() * 5) << "' class='row-header'>Dynamic Simulators (Heuristic)</td></tr>";
    write_solver_rows(dynamic_heuristic_solvers);

    report_file << "<tr><td colspan='" << (1 + scenarios.size() * 5) << "' class='row-header'>Dynamic Simulators (Advanced Heuristic)</td></tr>";
    write_solver_rows(advanced_heuristic_solvers);

    report_file << "<tr><td colspan='" << (1 + scenarios.size() * 5) << "' class='row-header'>Dynamic Simulators (RL-Based)</td></tr>";
    write_solver_rows(dynamic_rl_solvers);

    report_file << "<tr><td colspan='" << (1 + scenarios.size() * 5) << "' class='row-header'>Dynamic Simulators (Deep RL)</td></tr>";
    write_solver_rows(deep_rl_solvers);

    report_file << "<tr><td colspan='" << (1 + scenarios.size() * 5) << "' class='row-header'>EnMod-DP Hybrid Solvers</td></tr>";
    write_solver_rows(hybrid_solvers);

    report_file << "<tr><td colspan='" << (1 + scenarios.size() * 5) << "' class='row-header'>CUDA / GPU Accelerated Solvers</td></tr>";
    write_solver_rows(cuda_solvers);

    report_file << "</tbody></table>\n";
    writeHtmlFooter(report_file);
}