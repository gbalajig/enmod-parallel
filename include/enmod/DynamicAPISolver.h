#ifndef ENMOD_DYNAMIC_API_SOLVER_H
#define ENMOD_DYNAMIC_API_SOLVER_H

#include "DynamicSolver.h"
#include "Types.h" // <-- The critical fix is adding this line

class DynamicAPISolver : public Solver {
public:
    DynamicAPISolver(const Grid& grid_ref);
    void run() override;
    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;

private:
    std::vector<StepReport> history;
    Cost total_cost;
    EvacuationMode current_mode;

    void assessThreatAndSetMode(const Position& current_pos, const Grid& current_grid);
};

#endif // ENMOD_DYNAMIC_API_SOLVER_H

