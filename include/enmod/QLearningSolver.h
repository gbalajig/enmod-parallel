#ifndef ENMOD_Q_LEARNING_SOLVER_H
#define ENMOD_Q_LEARNING_SOLVER_H

#include "RLSolver.h"

class QLearningSolver : public RLSolver {
public:
    QLearningSolver(const Grid& grid_ref, const std::string& name = "QLearning");
    void run() override; // This will be the training step
    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;
    
    void train(int episodes) override;
    Direction chooseAction(const Position& state) override;
    void update(const Position& s, Direction a, double r, const Position& s_next, Direction a_next) override;
};

#endif // ENMOD_Q_LEARNING_SOLVER_H