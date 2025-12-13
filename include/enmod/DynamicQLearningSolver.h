#ifndef ENMOD_DYNAMIC_Q_LEARNING_SOLVER_H
#define ENMOD_DYNAMIC_Q_LEARNING_SOLVER_H

#include "QLearningSolver.h"
#include "DynamicSolver.h"

class DynamicQLearningSolver : public QLearningSolver {
public:
    DynamicQLearningSolver(const Grid& grid_ref);
    void run() override;
    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;

private:
    std::vector<StepReport> history;
    Cost total_cost;
};

#endif // ENMOD_DYNAMIC_Q_LEARNING_SOLVER_H

