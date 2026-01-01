#include "enmod/Logger.h"
#include "enmod/ScenarioGenerator.h"
#include "enmod/Grid.h"
#include "enmod/Solver.h"
#include "enmod/HtmlReportGenerator.h"

// --- 1. Static Solvers ---
#include "enmod/BIDP.h"
#include "enmod/FIDP.h"
#include "enmod/AStarSolver.h" 

// --- 2. Dynamic Solvers ---
#include "enmod/DynamicBIDPSolver.h"
#include "enmod/DynamicFIDPSolver.h"
#include "enmod/DynamicAVISolver.h" 
#include "enmod/DStarLiteSolver.h"

// --- 3. EnMod-DP Hybrid Solvers (Serial CPU) ---
#include "enmod/InterlacedSolver.h"
#include "enmod/HybridDPRLSolver.h"
#include "enmod/AdaptiveCostSolver.h"
#include "enmod/HierarchicalSolver.h"
#include "enmod/PolicyBlendingSolver.h"

// --- 4. Parallel Solvers (CUDA GPU) ---
#include "enmod/ParallelBIDP.h"
#include "enmod/ParallelStaticSolvers.h" 
#include "enmod/ParallelDynamicBIDPSolver.h"
#include "enmod/ParallelDynamicAVISolver.h"
#include "enmod/ParallelHybridSolvers.h" 

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
#include <thread>

bool isGpuAvailable() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess && deviceCount > 0);
}

// Updated Parameter: 'generate_detailed_html' controls the massive per-solver files
void runComparisonScenario(const json& config, const std::string& report_path, std::vector<Result>& results, bool generate_detailed_html) {
    Grid grid(config);
    
    // Only generate the initial map visual if detailed reports are ON
    if (generate_detailed_html) {
        HtmlReportGenerator::generateInitialGridReport(grid, report_path);
    }

    std::vector<std::unique_ptr<Solver>> solvers;

    // --- 1. Static ---
    solvers.push_back(std::make_unique<FIDP>(grid));   
    solvers.push_back(std::make_unique<BIDP>(grid));   
    solvers.push_back(std::make_unique<AStarSolver>(grid)); 

    // --- 2. Dynamic ---
    solvers.push_back(std::make_unique<DynamicBIDPSolver>(grid));
    solvers.push_back(std::make_unique<DStarLiteSolver>(grid));   

    // --- 3. EnMod-DP Serial ---
    solvers.push_back(std::make_unique<InterlacedSolver>(grid));       
    solvers.push_back(std::make_unique<HybridDPRLSolver>(grid));       
    solvers.push_back(std::make_unique<AdaptiveCostSolver>(grid));     
    solvers.push_back(std::make_unique<HierarchicalSolver>(grid));     
    solvers.push_back(std::make_unique<PolicyBlendingSolver>(grid));   

    // --- 4. Parallel ---
    if (isGpuAvailable()) {
        solvers.push_back(std::make_unique<ParallelFIDP>(grid)); 
        solvers.push_back(std::make_unique<ParallelBIDP>(grid)); 
        solvers.push_back(std::make_unique<ParallelAVI>(grid)); 
        solvers.push_back(std::make_unique<ParallelDynamicBIDPSolver>(grid));
        solvers.push_back(std::make_unique<ParallelDynamicAVISolver>(grid));
        solvers.push_back(std::make_unique<ParallelInterlacedSolver>(grid));
        solvers.push_back(std::make_unique<ParallelHybridDPRLSolver>(grid));
        solvers.push_back(std::make_unique<ParallelAdaptiveCostSolver>(grid));
        solvers.push_back(std::make_unique<ParallelHierarchicalSolver>(grid));
        solvers.push_back(std::make_unique<ParallelPolicyBlendingSolver>(grid));
    }

    for (const auto& solver : solvers) {
        if (isGpuAvailable()) cudaDeviceSynchronize();
        auto start_time = std::chrono::high_resolution_clock::now();
        
        solver->run();
        
        if (isGpuAvailable()) cudaDeviceSynchronize();
        auto end_time = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> execution_time = end_time - start_time;

        Cost final_cost = solver->getEvacuationCost();
        double weighted_cost = (final_cost.distance == MAX_COST) 
                               ? std::numeric_limits<double>::infinity() 
                               : (final_cost.smoke * 1000) + (final_cost.time * 10) + (final_cost.distance * 1);
        
        results.push_back({grid.getName(), solver->getName(), final_cost, weighted_cost, execution_time.count()});
        
        // --- CRITICAL SPACE SAVER ---
        // Only generate the heavy HTML report if specifically requested
        if (generate_detailed_html) {
            HtmlReportGenerator::generateSolverReport(*solver, report_path);
        }
    }
}

int main() {
    try {
        std::filesystem::create_directory("reports");
        Logger::init("logs/enmod_simulation.log");
        
        std::stringstream ss;
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");
        std::string master_root_path = "reports/Benchmark_1500Exp_" + ss.str();
        std::filesystem::create_directory(master_root_path);

        std::cout << "Log file created at: logs/enmod_simulation.log\n";
        std::cout << "Reports will be generated in: " << master_root_path << "\n";

        // --- Config ---
        std::vector<int> grid_sizes = {10, 25, 50,100};
        std::vector<double> densities = {0.0, 0.15, 0.5};
        int num_trials = 3;

        // *** SPACE SAVING CONFIGURATION ***
        // FALSE = Do not generate ~30,000 detailed HTML files (Saves GBs of space)
        // TRUE  = Generate everything (Will likely crash on 128x128)
        bool generate_detailed_html = false; 

        std::cout << "Starting Large-Scale Benchmark (1500 Experiments)...\n";
        std::cout << "Detailed HTML Reports: " << (generate_detailed_html ? "ENABLED" : "DISABLED (Summary Only)") << "\n";

        std::vector<Result> global_results; 

        // --- Loop 1: Grid Size ---
        for (int size : grid_sizes) {
            std::string size_folder = master_root_path + "/" + std::to_string(size) + "x" + std::to_string(size);
            std::filesystem::create_directory(size_folder);
            
            std::vector<Result> size_results; // Aggregates results for this grid size

            // --- Loop 2: Density ---
            for (double density : densities) {
                // --- Loop 3: Random Trials ---
                for (int trial = 1; trial <= num_trials; ++trial) {
                    std::cout << "Running " << size << "x" << size << " | D: " << density << " | Trial " << trial << "... " << std::flush;
                    
                    // Create Per-Trial Folder (Organization)
                    std::string trial_folder_name = "D" + std::to_string((int)(density*100)) + "_T" + std::to_string(trial);
                    std::string full_trial_path = size_folder + "/" + trial_folder_name;
                    std::filesystem::create_directory(full_trial_path);

                    std::string name = std::to_string(size) + "x" + std::to_string(size) + "_" + trial_folder_name;
                    
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    json config = ScenarioGenerator::generate(size, name, density);

                    std::vector<Result> trial_results;
                    
                    // Run Simulation (Detailed HTML is OFF)
                    runComparisonScenario(config, full_trial_path, trial_results, generate_detailed_html);

                    // 1. Generate Per-Trial Summary (Small file, ~2KB)
                    HtmlReportGenerator::generateSummaryReport(trial_results, full_trial_path, "_Summary_Report.html");

                    // 2. Accumulate Results
                    size_results.insert(size_results.end(), trial_results.begin(), trial_results.end());
                    global_results.insert(global_results.end(), trial_results.begin(), trial_results.end());

                    std::cout << "Done.\n";
                }
            }

            // 3. Generate Average Summary for this Grid Size
            std::cout << "Generating Average Report for " << size << "x" << size << "...\n";
            HtmlReportGenerator::generateSummaryReport(size_results, size_folder, "_Average_Summary_Report.html");
        }

        // 4. Generate Master Global Report (All 1500 experiments aggregated)
        std::cout << "Generating Master Global Report...\n";
        HtmlReportGenerator::generateSummaryReport(global_results, master_root_path, "_Global_Summary_Report.html");

        std::cout << "\nBenchmark complete.\n";
        Logger::close();
    } catch (const std::exception& e) {
        std::cerr << "Critical Error: " << e.what() << std::endl;
        Logger::log(LogLevel::ERROR, "Critical Error: " + std::string(e.what()));
        return 1;
    }
    return 0;
}