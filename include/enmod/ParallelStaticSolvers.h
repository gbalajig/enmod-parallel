#ifndef ENMOD_PARALLEL_STATIC_SOLVERS_H
#define ENMOD_PARALLEL_STATIC_SOLVERS_H

#include "ParallelSolverBase.h"

// --- ParallelFIDP ---
class ParallelFIDP : public ParallelSolverBase {
public:
    ParallelFIDP(const Grid& grid);
    void run() override;
    Cost getEvacuationCost() const override;
};

// --- ParallelAVI ---
class ParallelAVI : public ParallelSolverBase {
public:
    ParallelAVI(const Grid& grid);
    void run() override;
};

#endif // ENMOD_PARALLEL_STATIC_SOLVERS_H