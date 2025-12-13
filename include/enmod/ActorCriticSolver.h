#ifndef ENMOD_ACTOR_CRITIC_SOLVER_H
#define ENMOD_ACTOR_CRITIC_SOLVER_H

#include "RLSolver.h"

class ActorCriticSolver : public RLSolver {
public:
    ActorCriticSolver(const Grid& grid_ref);
    void run() override;
    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;
    
    void update(const Position& s, Direction a, double r, const Position& s_next, Direction a_next) override;
    Direction chooseAction(const Position& state) override;

private:
    // The Critic's state-value table
    std::map<Position, double> state_value_table;
};

#endif // ENMOD_ACTOR_CRITIC_SOLVER_H

