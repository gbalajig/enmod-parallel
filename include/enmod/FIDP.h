#ifndef ENMOD_FIDP_H
#define ENMOD_FIDP_H

#include "Solver.h"
#include <vector>

class FIDP : public Solver {
public:
    FIDP(const Grid& grid_ref);
    
    void run() override;
    void run(const Position& start_pos); 

    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;
    std::vector<Position> getEvacuationPath(const Position& start_pos) const;

private:
    std::vector<std::vector<Cost>> cost_map;
    std::vector<std::vector<Position>> parent_map;
};

#endif // ENMOD_FIDP_H

