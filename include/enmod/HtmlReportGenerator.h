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
    static void generateSummaryReport(const std::vector<Result>& results, const std::string& path);
};

#endif // ENMOD_HTML_REPORT_GENERATOR_H