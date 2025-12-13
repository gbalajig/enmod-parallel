#ifndef ENMOD_POLICY_VERIFIER_H
#define ENMOD_POLICY_VERIFIER_H

#include "Solver.h"
#include "Policy.h"
#include <vector>

class PolicyVerifier : public Solver {
public:
    PolicyVerifier(const Grid& grid_ref);
    void run() override;
    Cost getEvacuationCost() const override;
    void generateReport(std::ofstream& report_file) const override;

private:
    bool is_regular;
    bool isPathRegular(const Position& start_pos, const Policy& policy) const;
};

#endif // ENMOD_POLICY_VERIFIER_H

