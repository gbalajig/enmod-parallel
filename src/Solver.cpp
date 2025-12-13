#include "enmod/Solver.h"

Solver::Solver(const Grid& grid_ref, const std::string& name) 
    : grid(grid_ref), solver_name(name) {}

const std::string& Solver::getName() const {
    return solver_name;
}

