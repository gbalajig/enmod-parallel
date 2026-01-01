#include "enmod/HtmlReportGenerator.h"
#include <fstream>
#include <map>
#include <algorithm>
#include <vector>
#include <iomanip>
#include <iostream>

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

// Updated to use the filename parameter
void HtmlReportGenerator::generateSummaryReport(const std::vector<Result>& results, const std::string& path, const std::string& filename) {
    std::string file_path = path + "/" + filename;
    std::ofstream report_file(file_path);
    if (!report_file) return;

    writeHtmlHeader(report_file, "Simulation Summary Report");
    report_file << "<h1>Simulation Summary (" << filename << ")</h1>\n";

    struct Stats {
        int count = 0;
        int failures = 0;
        long long smoke = 0;
        long long time = 0;
        long long dist = 0;
        double w_cost = 0;
        double exec_time = 0;
    };

    std::map<std::string, std::map<std::string, Stats>> aggregated_data;
    std::vector<std::string> base_scenarios;

    for (const auto& res : results) {
        // Intelligent aggregation:
        // If results contain multiple trials (e.g. "_T1", "_T2"), this logic groups them by base name.
        // If results are from a single trial, "base_name" is just the scenario name, and average = actual value.
        std::string base_name = res.scenario_name;
        size_t last_underscore = base_name.find_last_of('_');
        // Check if suffix is like "_T1", "_T2"
        if (last_underscore != std::string::npos && base_name.substr(last_underscore).rfind("_T", 0) == 0) {
            base_name = base_name.substr(0, last_underscore);
        }

        if (aggregated_data.find(base_name) == aggregated_data.end()) {
            base_scenarios.push_back(base_name);
        }

        Stats& s = aggregated_data[base_name][res.solver_name];
        s.count++;
        
        if (res.cost.distance >= MAX_COST || res.cost.distance < 0) {
            s.failures++;
        } else {
            s.smoke += res.cost.smoke;
            s.time += res.cost.time;
            s.dist += res.cost.distance;
            s.w_cost += res.weighted_cost;
            s.exec_time += res.execution_time;
        }
    }

    std::sort(base_scenarios.begin(), base_scenarios.end(), [](const std::string& a, const std::string& b){
        try {
            int size_a = std::stoi(a.substr(0, a.find('x')));
            int size_b = std::stoi(b.substr(0, b.find('x')));
            if (size_a != size_b) return size_a < size_b;
        } catch(...) {}
        return a < b;
    });

    // --- Dynamic Solver List Construction ---
    // Instead of hardcoding, collect all unique solver names found in results
    std::vector<std::string> all_solvers;
    for (const auto& pair : aggregated_data.begin()->second) {
        all_solvers.push_back(pair.first);
    }
    // Sort broadly to keep Serial/Parallel grouped if named consistently
    std::sort(all_solvers.begin(), all_solvers.end());

    report_file << "<table>\n";
    report_file << "<thead><tr><th rowspan='2'>Algorithm</th>";
    for(const auto& scn : base_scenarios){
        report_file << "<th colspan='5'>" << scn << "</th>";
    }
    report_file << "</tr>\n";

    report_file << "<tr>";
    for(size_t i = 0; i < base_scenarios.size(); ++i){
        report_file << "<th>Smoke</th><th>Time</th><th>Dist</th><th>W. Cost</th><th>Exec(ms)</th>";
    }
    report_file << "</tr></thead>\n<tbody>";

    for(const auto& solver : all_solvers){
        report_file << "<tr><td>" << solver << "</td>";
        for(const auto& scn : base_scenarios){
            if (aggregated_data.find(scn) == aggregated_data.end() || 
                aggregated_data[scn].find(solver) == aggregated_data[scn].end()) {
                report_file << "<td colspan='5' style='color:gray'>Not Run</td>";
                continue;
            }

            const auto& stats = aggregated_data[scn][solver];
            int successful_runs = stats.count - stats.failures;

            if (successful_runs > 0) {
                report_file << "<td>" << (stats.smoke / successful_runs) << "</td>";
                report_file << "<td>" << (stats.time / successful_runs) << "</td>";
                report_file << "<td>" << (stats.dist / successful_runs) << "</td>";
                report_file << "<td>" << static_cast<long long>(stats.w_cost / successful_runs) << "</td>";
                report_file << "<td>" << std::fixed << std::setprecision(2) << (stats.exec_time / successful_runs) << "</td>";
            } else if (stats.count > 0) {
                report_file << "<td colspan='5' style='color:red'>ALL FAILED</td>";
            } else {
                report_file << "<td colspan='5' style='color:gray'>Not Run</td>";
            }
        }
        report_file << "</tr>\n";
    }

    report_file << "</tbody></table>\n";
    writeHtmlFooter(report_file);
}