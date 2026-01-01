#ifndef ENMOD_SCENARIO_GENERATOR_H
#define ENMOD_SCENARIO_GENERATOR_H

#include "json.hpp"
#include "Types.h"
#include <string>

using json = nlohmann::json;

class ScenarioGenerator {
public:
    // Update: Added 'obstacle_density' with a default value of 0.0
    static json generate(int size, const std::string& name, double obstacle_density = 0.0);
};

#endif // ENMOD_SCENARIO_GENERATOR_H