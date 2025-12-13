#ifndef ENMOD_PARALLEL_DYNAMIC_SOLVERS_H
#define ENMOD_PARALLEL_DYNAMIC_SOLVERS_H

#include "DynamicSolver.h"
#include "Types.h"

// Abstract base to share the common loop logic
class ParallelDynamicSolverHelper : public Solver {
public:
    ParallelDynamicSolverHelper(const Grid& grid_ref, const std::string& name);
    void run() override;
    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;

protected:
    // Children implement this to run their specific GPU algo
    virtual const std::vector<std::vector<Cost>>& run_gpu_step(const Grid& step_grid) = 0;

    std::vector<StepReport> history;
    Cost total_cost;
    EvacuationMode current_mode;
    void assessThreatAndSetMode(const Position& current_pos, const Grid& current_grid);
};

class ParallelDynamicFIDPSolver : public ParallelDynamicSolverHelper {
public:
    ParallelDynamicFIDPSolver(const Grid& grid);
protected:
    const std::vector<std::vector<Cost>>& run_gpu_step(const Grid& step_grid) override;
};

class ParallelDynamicAVISolver : public ParallelDynamicSolverHelper {
public:
    ParallelDynamicAVISolver(const Grid& grid);
protected:
    const std::vector<std::vector<Cost>>& run_gpu_step(const Grid& step_grid) override;
};

#endif // ENMOD_PARALLEL_DYNAMIC_SOLVERS_H