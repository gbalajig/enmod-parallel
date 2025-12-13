#ifndef ENMOD_API_H
#define ENMOD_API_H

#include "Solver.h"
#include "Policy.h"
#include <vector>

class API : public Solver {
public:
    API(const Grid& grid_ref);
    void run() override;
    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;
    
    const Policy& getPolicy() const;

private:
    Cost calculatePolicyCost() const;
    Policy policy;
};

#endif // ENMOD_API_H

