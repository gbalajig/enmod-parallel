#ifndef ENMOD_AVI_H
#define ENMOD_AVI_H

#include "Solver.h"
#include <vector>

class AVI : public Solver {
public:
    AVI(const Grid& grid_ref);
    void run() override;
    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;
    const std::vector<std::vector<Cost>>& getCostMap() const;

private:
    std::vector<std::vector<Cost>> cost_map;
};

#endif // ENMOD_AVI_H

