#ifndef ENMOD_POLICY_GENERATOR_H
#define ENMOD_POLICY_GENERATOR_H

#include "Solver.h"
#include "Policy.h"

class PolicyGenerator : public Solver {
public:
    PolicyGenerator(const Grid& grid_ref);
    void run() override;
    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;
    const Policy& getPolicy() const;

private:
    Cost calculatePolicyCost() const;
    Policy policy;
    Cost final_cost;
};

#endif // ENMOD_POLICY_GENERATOR_H

