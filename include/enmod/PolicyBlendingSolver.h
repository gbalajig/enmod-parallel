#ifndef ENMOD_POLICY_BLENDING_SOLVER_H
#define ENMOD_POLICY_BLENDING_SOLVER_H

#include "DynamicSolver.h"
#include "Types.h"
#include "QLearningSolver.h"
#include <memory>

class PolicyBlendingSolver : public Solver {
public:
    PolicyBlendingSolver(const Grid& grid_ref);
    void run() override;
    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;

private:
    std::vector<StepReport> history;
    Cost total_cost;
    EvacuationMode current_mode;
    std::unique_ptr<QLearningSolver> rl_solver;

    void assessThreatAndSetMode(const Position& current_pos, const Grid& current_grid);
};

#endif // ENMOD_POLICY_BLENDING_SOLVER_H