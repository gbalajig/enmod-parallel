#ifndef ENMOD_DYNAMIC_ACTOR_CRITIC_SOLVER_H
#define ENMOD_DYNAMIC_ACTOR_CRITIC_SOLVER_H

#include "ActorCriticSolver.h"
#include "DynamicSolver.h"

class DynamicActorCriticSolver : public ActorCriticSolver {
public:
    DynamicActorCriticSolver(const Grid& grid_ref);
    void run() override;
    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;

private:
    std::vector<StepReport> history;
    Cost total_cost;
};

#endif // ENMOD_DYNAMIC_ACTOR_CRITIC_SOLVER_H

