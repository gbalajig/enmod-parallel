#include "enmod/Logger.h"
#include "enmod/ScenarioGenerator.h"
#include "enmod/Grid.h"
#include "enmod/Solver.h"
#include "enmod/HtmlReportGenerator.h"
// Static DP Solvers
#include "enmod/BIDP.h"
#include "enmod/FIDP.h"
#include "enmod/API.h"
// Dynamic DP Solvers
#include "enmod/DynamicBIDPSolver.h"
#include "enmod/DynamicAPISolver.h"
#include "enmod/DynamicFIDPSolver.h"
#include "enmod/DynamicAVISolver.h"
// RL Solvers
#include "enmod/QLearningSolver.h"
#include "enmod/SARSASolver.h"
#include "enmod/ActorCriticSolver.h"
#include "enmod/DynamicQLearningSolver.h"
#include "enmod/DynamicSARSASolver.h"
#include "enmod/DynamicActorCriticSolver.h"
// EnMod-DP Solvers
#include "enmod/HybridDPRLSolver.h"
#include "enmod/AdaptiveCostSolver.h"
#include "enmod/InterlacedSolver.h"
#include "enmod/HierarchicalSolver.h"
#include "enmod/PolicyBlendingSolver.h"
// Multi-Agent CPS
#include "enmod/MultiAgentCPSController.h"
// Parallel Solvers
#include "enmod/ParallelBIDP.h"
#include "enmod/ParallelStaticSolvers.h" // FIDP, AVI
#include "enmod/ParallelDynamicBIDPSolver.h"
#include "enmod/ParallelDynamicSolvers.h" // Dynamic FIDP, AVI
#include "enmod/ParallelHybridSolvers.h" // Hybrid, Interlaced, Hierarchical, PolicyBlending, Adaptive
#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <memory>
#include <filesystem>
#include <map>
#include <chrono>
#include <sstream>

#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif

bool isGpuAvailable() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) return false;
    return deviceCount > 0;
}

void runComparisonScenario(const json& config, const std::string& report_path, std::vector<Result>& results) {
    Grid grid(config);
    std::cout << "\n===== Running Comparison Scenario: " << grid.getName() << " (" << grid.getRows() << "x" << grid.getCols() << ") =====\n";
    std::string scenario_report_path = report_path + "/" + grid.getName();
    std::filesystem::create_directory(scenario_report_path);
    HtmlReportGenerator::generateInitialGridReport(grid, scenario_report_path);

    std::vector<std::unique_ptr<Solver>> solvers;

    // --- Standard Solvers ---
    solvers.push_back(std::make_unique<BIDP>(grid));
    solvers.push_back(std::make_unique<FIDP>(grid));
    solvers.push_back(std::make_unique<API>(grid));
    solvers.push_back(std::make_unique<QLearningSolver>(grid));
    solvers.push_back(std::make_unique<SARSASolver>(grid));
    solvers.push_back(std::make_unique<ActorCriticSolver>(grid));
    solvers.push_back(std::make_unique<DynamicBIDPSolver>(grid));
    solvers.push_back(std::make_unique<DynamicFIDPSolver>(grid));
    solvers.push_back(std::make_unique<DynamicAVISolver>(grid));
    solvers.push_back(std::make_unique<DynamicAPISolver>(grid));
    solvers.push_back(std::make_unique<DynamicQLearningSolver>(grid));
    solvers.push_back(std::make_unique<DynamicSARSASolver>(grid));
    solvers.push_back(std::make_unique<DynamicActorCriticSolver>(grid));
    solvers.push_back(std::make_unique<HybridDPRLSolver>(grid));
    solvers.push_back(std::make_unique<AdaptiveCostSolver>(grid));
    solvers.push_back(std::make_unique<InterlacedSolver>(grid));
    solvers.push_back(std::make_unique<HierarchicalSolver>(grid));
    solvers.push_back(std::make_unique<PolicyBlendingSolver>(grid));
    
    // --- CUDA / GPU Solvers ---
    if (isGpuAvailable()) {
        std::cout << "[INFO] GPU detected. Enabling Parallel Solvers.\n";
        // Static
        solvers.push_back(std::make_unique<ParallelBIDP>(grid));
        solvers.push_back(std::make_unique<ParallelFIDP>(grid));
        solvers.push_back(std::make_unique<ParallelAVI>(grid));
        
        // Dynamic
        solvers.push_back(std::make_unique<ParallelDynamicBIDPSolver>(grid));
        solvers.push_back(std::make_unique<ParallelDynamicFIDPSolver>(grid));
        solvers.push_back(std::make_unique<ParallelDynamicAVISolver>(grid));

        // Hybrid
        solvers.push_back(std::make_unique<ParallelHybridDPRLSolver>(grid));
        solvers.push_back(std::make_unique<ParallelAdaptiveCostSolver>(grid));
        solvers.push_back(std::make_unique<ParallelInterlacedSolver>(grid));
        solvers.push_back(std::make_unique<ParallelHierarchicalSolver>(grid));
        solvers.push_back(std::make_unique<ParallelPolicyBlendingSolver>(grid));
    } else {
        std::cout << "[WARN] No CUDA-capable GPU detected. Skipping Parallel Solvers.\n";
        Logger::log(LogLevel::WARN, "Skipping GPU solvers: No device found.");
    }

    for (const auto& solver : solvers) {
        std::cout << "  - Running " << solver->getName() << "... ";
        auto start_time = std::chrono::high_resolution_clock::now();
        solver->run();
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> execution_time = end_time - start_time;
        std::cout << "Done (" << std::fixed << std::setprecision(2) << execution_time.count() << " ms).\n";

        Cost final_cost = solver->getEvacuationCost();
        double weighted_cost = (final_cost.distance == MAX_COST) ? std::numeric_limits<double>::infinity() : (final_cost.smoke * 1000) + (final_cost.time * 10) + (final_cost.distance * 1);
        results.push_back({grid.getName(), solver->getName(), final_cost, weighted_cost, execution_time.count()});
        HtmlReportGenerator::generateSolverReport(*solver, scenario_report_path);
    }
}

int main() {
    try {
        std::filesystem::create_directory("logs");
        std::filesystem::create_directory("reports");
        Logger::init("logs/enmod_simulation.log");
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");
        std::string report_root_path = "reports/run_" + ss.str();
        std::filesystem::create_directory(report_root_path);
        std::cout << "Log file created at: logs/enmod_simulation.log\n";
        std::cout << "Reports will be generated in: " << report_root_path << "\n";

        std::vector<json> scenarios;
        scenarios.push_back(ScenarioGenerator::generate(5, "5x5"));
        scenarios.push_back(ScenarioGenerator::generate(10, "10x10"));
        scenarios.push_back(ScenarioGenerator::generate(15, "15x15"));

        std::vector<Result> all_results;
        for (const auto& config : scenarios) {
            runComparisonScenario(config, report_root_path, all_results);
        }

        HtmlReportGenerator::generateSummaryReport(all_results, report_root_path);
        std::cout << "\nComparison simulation complete. Summary written to " << report_root_path << "/_Summary_Report.html\n";

        std::cout << "\n--- Starting Multi-Agent Simulation with HybridDPRLSolver ---\n";
        for(const auto& config : scenarios) {
            MultiAgentCPSController cps_controller(config, report_root_path + "/" + config["name"].get<std::string>(), 5);
            cps_controller.run_simulation();
        }

        Logger::log(LogLevel::INFO, "Application Finished Successfully");
        Logger::close();
    } catch (const std::exception& e) {
        std::cerr << "A critical error occurred: " << e.what() << std::endl;
        Logger::log(LogLevel::ERROR, "A critical error occurred: " + std::string(e.what()));
        return 1;
    }
    return 0;
}