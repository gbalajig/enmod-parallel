#ifndef ENMOD_RL_SOLVER_H
#define ENMOD_RL_SOLVER_H

#include "Solver.h"
#include "Policy.h"
#include <vector>
#include <map>

using ValueTable = std::map<Position, std::vector<double>>;

class RLSolver : public Solver {
public:
    RLSolver(const Grid& grid_ref, const std::string& name);
    
    void run() override; // Main training loop for static solvers

    // RL-specific methods for learning
    virtual void update(const Position& s, Direction a, double r, const Position& s_next, Direction a_next) = 0;
    virtual Direction chooseAction(const Position& state) = 0;
    const Policy& getPolicy(); 
    const ValueTable& getPolicyValueTable() const { return value_table; }

protected:
    void generatePolicyFromValueTable();
    virtual void train(int episodes); 

    ValueTable value_table;
    Policy policy;
    
    double alpha = 0.1;
    double gamma = 0.9;
    double epsilon = 0.1;
};

#endif // ENMOD_RL_SOLVER_H