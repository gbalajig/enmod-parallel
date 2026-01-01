#ifndef ENMOD_HTML_REPORT_GENERATOR_H
#define ENMOD_HTML_REPORT_GENERATOR_H

#include "Grid.h"
#include "Solver.h"
#include <string>
#include <vector>

struct Result {
    std::string scenario_name;
    std::string solver_name;
    Cost cost;
    double weighted_cost;
    double execution_time; 
};

class HtmlReportGenerator {
public:
    static void generateInitialGridReport(const Grid& grid, const std::string& path);
    static void generateSolverReport(const Solver& solver, const std::string& path);
    
    // Updated: Added 'filename' parameter with a default value
    static void generateSummaryReport(const std::vector<Result>& results, const std::string& path, const std::string& filename = "_Summary_Report.html");
};

#endif // ENMOD_HTML_REPORT_GENERATOR_H