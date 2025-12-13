#include "enmod/ParallelBIDP.h"

ParallelBIDP::ParallelBIDP(const Grid& grid) : ParallelSolverBase(grid, "ParallelBIDP") {}

void ParallelBIDP::run() {
    // BIDP (Backward IDP) initializes exits with 0 cost
    run_cuda_algo(grid.getExitPositions());
}