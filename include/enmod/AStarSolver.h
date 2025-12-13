#ifndef ENMOD_ASTAR_SOLVER_H
#define ENMOD_ASTAR_SOLVER_H

#include "Solver.h"
#include <vector>

class AStarSolver : public Solver {
public:
    AStarSolver(const Grid& grid_ref);
    void run() override;
    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;

private:
    std::vector<Position> path;
    Cost total_cost;
};

#endif // ENMOD_ASTAR_SOLVER_H