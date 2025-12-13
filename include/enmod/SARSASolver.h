#ifndef ENMOD_SARSA_SOLVER_H
#define ENMOD_SARSA_SOLVER_H

#include "RLSolver.h"

class SARSASolver : public RLSolver {
public:
    SARSASolver(const Grid& grid_ref);
    void run() override;
    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;
    
    void train(int episodes) override;
    Direction chooseAction(const Position& state) override;
    void update(const Position& s, Direction a, double r, const Position& s_next, Direction a_next) override;
};

#endif // ENMOD_SARSA_SOLVER_H