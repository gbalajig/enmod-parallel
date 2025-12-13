#ifndef ENMOD_BIDP_H
#define ENMOD_BIDP_H

#include "Solver.h"
#include <vector>

class BIDP : public Solver {
public:
    BIDP(const Grid& grid_ref);
    void run() override;
    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;
    
    // Expose the cost map for dynamic solvers
    const std::vector<std::vector<Cost>>& getCostMap() const;

private:
    std::vector<std::vector<Cost>> cost_map;
};

#endif // ENMOD_BIDP_H

