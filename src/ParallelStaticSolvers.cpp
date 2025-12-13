#include "enmod/ParallelStaticSolvers.h"
#include <algorithm>

// --- ParallelFIDP ---
ParallelFIDP::ParallelFIDP(const Grid& grid) : ParallelSolverBase(grid, "ParallelFIDP") {}

void ParallelFIDP::run() {
    // FIDP (Forward IDP) initializes Start with 0 cost and propagates
    run_cuda_algo({grid.getStartPosition()});
}

Cost ParallelFIDP::getEvacuationCost() const {
    // For FIDP, the evacuation cost is the cost at the BEST exit
    Cost best_cost = {}; // Default MAX_COST
    for(const auto& exit : grid.getExitPositions()) {
        if (grid.isValid(exit.row, exit.col)) {
            Cost c = cost_map[exit.row][exit.col];
            if (c.distance != 2147483647) {
                // Simple comparison logic
                if (best_cost.distance == 2147483647 || c.distance < best_cost.distance) {
                    best_cost = c;
                }
            }
        }
    }
    return best_cost;
}

// --- ParallelAVI ---
ParallelAVI::ParallelAVI(const Grid& grid) : ParallelSolverBase(grid, "ParallelAVI") {}

void ParallelAVI::run() {
    // AVI (Value Iteration) initializes Exits with 0 cost (like BIDP)
    run_cuda_algo(grid.getExitPositions());
}