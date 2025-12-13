#ifndef ENMOD_ADA_SOLVER_H
#define ENMOD_ADA_SOLVER_H

#include "DynamicSolver.h"
#include "Types.h"
#include <vector>

class ADASolver : public Solver {
public:
    ADASolver(const Grid& grid_ref);
    void run() override;
    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;

private:
    std::vector<StepReport> history;
    Cost total_cost;
    EvacuationMode current_mode;
    double epsilon; // Inflation factor

    void assessThreatAndSetMode(const Position& current_pos, const Grid& current_grid);
};

#endif // ENMOD_ADA_SOLVER_H