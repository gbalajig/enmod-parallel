#ifndef ENMOD_DSTARLITE_SOLVER_H
#define ENMOD_DSTARLITE_SOLVER_H

#include "DynamicSolver.h"
#include "Types.h"
#include <vector>
#include <map>
#include <queue>

// Define a pair for the priority key in D* Lite
using Key = std::pair<double, double>;

struct DStarNode {
    double g = std::numeric_limits<double>::infinity();
    double rhs = std::numeric_limits<double>::infinity();
};

class DStarLiteSolver : public Solver {
public:
    DStarLiteSolver(const Grid& grid_ref);
    void run() override;
    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;

private:
    void initialize();
    double heuristic(const Position& a, const Position& b);
    Key calculateKey(const Position& p);
    void updateVertex(const Position& p);
    void computeShortestPath();
    Grid dynamic_grid; // Make grid a member to be accessible in all functions

    std::vector<StepReport> history;
    Cost total_cost;
    Position start_pos;
    Position goal_pos;
    Position last_pos = {-1, -1};
    double k_m = 0.0;
    std::map<Position, DStarNode> maze;
    std::priority_queue<std::pair<Key, Position>, std::vector<std::pair<Key, Position>>, std::greater<std::pair<Key, Position>>> open_set;
};

#endif // ENMOD_DSTARLITE_SOLVER_H