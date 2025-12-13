#ifndef ENMOD_PARALLEL_BIDP_H
#define ENMOD_PARALLEL_BIDP_H

#include "ParallelSolverBase.h"

class ParallelBIDP : public ParallelSolverBase {
public:
    ParallelBIDP(const Grid& grid);
    void run() override;
};

#endif // ENMOD_PARALLEL_BIDP_H