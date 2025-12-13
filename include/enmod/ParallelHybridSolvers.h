#ifndef ENMOD_PARALLEL_HYBRID_SOLVERS_H
#define ENMOD_PARALLEL_HYBRID_SOLVERS_H

#include "DynamicSolver.h"
#include "QLearningSolver.h"
#include <memory>

// 1. Parallel Hybrid DP-RL
class ParallelHybridDPRLSolver : public Solver {
public:
    ParallelHybridDPRLSolver(const Grid& grid_ref);
    void run() override;
    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;
private:
    std::unique_ptr<QLearningSolver> rl_solver;
    std::vector<StepReport> history;
    Cost total_cost;
    void assessThreatAndSetMode(const Position& current_pos, const Grid& current_grid);
};

// 2. Parallel Adaptive Cost
class ParallelAdaptiveCostSolver : public Solver {
public:
    ParallelAdaptiveCostSolver(const Grid& grid_ref);
    void run() override;
    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;
private:
    std::vector<StepReport> history;
    Cost total_cost;
    void assessThreatAndSetMode(const Position& current_pos, const Grid& current_grid);
};

// 3. Parallel Interlaced
class ParallelInterlacedSolver : public Solver {
public:
    ParallelInterlacedSolver(const Grid& grid_ref);
    void run() override;
    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;
private:
    std::vector<StepReport> history;
    Cost total_cost;
    void assessThreatAndSetMode(const Position& current_pos, const Grid& current_grid);
};

// 4. Parallel Hierarchical
class ParallelHierarchicalSolver : public Solver {
public:
    ParallelHierarchicalSolver(const Grid& grid_ref);
    void run() override;
    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;
private:
    std::vector<StepReport> history;
    Cost total_cost;
    std::vector<Position> current_plan;
    void assessThreatAndSetMode(const Position& current_pos, const Grid& current_grid);
};

// 5. Parallel Policy Blending
class ParallelPolicyBlendingSolver : public Solver {
public:
    ParallelPolicyBlendingSolver(const Grid& grid_ref);
    void run() override;
    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;
private:
    std::unique_ptr<QLearningSolver> rl_solver;
    std::vector<StepReport> history;
    Cost total_cost;
    void assessThreatAndSetMode(const Position& current_pos, const Grid& current_grid);
};

#endif // ENMOD_PARALLEL_HYBRID_SOLVERS_H