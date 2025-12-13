#ifndef ENMOD_HYBRID_DP_RL_SOLVER_H
#define ENMOD_HYBRID_DP_RL_SOLVER_H

#include "DynamicSolver.h"
#include "Types.h"
#include "BIDP.h"
#include "QLearningSolver.h"
#include <memory>

class HybridDPRLSolver : public Solver {
public:
    HybridDPRLSolver(const Grid& grid_ref);
    void run() override;
    Direction getNextMove(const Position& current_pos, const Grid& current_grid); 
    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;

private:
    std::vector<StepReport> history;
    Cost total_cost;
    EvacuationMode current_mode;

    std::unique_ptr<QLearningSolver> rl_solver;

    void assessThreatAndSetMode(const Position& current_pos, const Grid& current_grid);
};

#endif // ENMOD_HYBRID_DP_RL_SOLVER_H