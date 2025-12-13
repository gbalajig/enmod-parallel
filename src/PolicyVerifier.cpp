#include "enmod/PolicyVerifier.h"
#include "enmod/PolicyGenerator.h"
#include "enmod/Logger.h"
#include <iostream>
#include <vector>
#include <set>

PolicyVerifier::PolicyVerifier(const Grid& grid_ref) 
    : Solver(grid_ref, "PolicyVerify"), is_regular(false) {}

void PolicyVerifier::run() {
    PolicyGenerator temp_policy_gen(grid);
    temp_policy_gen.run();
    const Policy& optimal_policy = temp_policy_gen.getPolicy();

    is_regular = true;
    for (int r = 0; r < grid.getRows(); ++r) {
        for (int c = 0; c < grid.getCols(); ++c) {
            if (grid.isWalkable(r, c)) {
                if (!isPathRegular({r, c}, optimal_policy)) {
                    is_regular = false;
                    Logger::log(LogLevel::ERROR, "Policy is IRREGULAR. Fails at starting point: (" + std::to_string(r) + "," + std::to_string(c) + ")");
                    return; 
                }
            }
        }
    }
}

bool PolicyVerifier::isPathRegular(const Position& start_pos, const Policy& policy) const {
    Position current = start_pos;
    std::set<Position> visited;
    for (int i = 0; i < grid.getRows() * grid.getCols() + 2; ++i) {
        if (grid.isExit(current.row, current.col)) {
            return true; 
        }
        if (visited.count(current)) {
            return false; 
        }
        visited.insert(current);
        Direction dir = policy.getDirection(current);
        current = grid.getNextPosition(current, dir);
        if (!grid.isWalkable(current.row, current.col)) {
             return false; 
        }
    }
    return false;
}

Cost PolicyVerifier::getEvacuationCost() const {
    return {};
}

void PolicyVerifier::generateReport(std::ofstream& report_file) const {
    report_file << "<h2>Policy Verification Result</h2>\n";
    if (is_regular) {
        report_file << "<p style='color:green; font-weight:bold;'>Policy is REGULAR: All paths from all walkable cells lead to an exit.</p>\n";
    } else {
        report_file << "<p style='color:red; font-weight:bold;'>Policy is IRREGULAR: Not all paths lead to an exit (check log for details).</p>\n";
    }
}

