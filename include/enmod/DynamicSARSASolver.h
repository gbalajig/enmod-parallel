#ifndef ENMOD_DYNAMIC_SARSA_SOLVER_H
#define ENMOD_DYNAMIC_SARSA_SOLVER_H

#include "SARSASolver.h"
#include "DynamicSolver.h"

class DynamicSARSASolver : public SARSASolver {
public:
    DynamicSARSASolver(const Grid& grid_ref);
    void run() override;
    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;

private:
    std::vector<StepReport> history;
    Cost total_cost;
};

#endif // ENMOD_DYNAMIC_SARSA_SOLVER_H

