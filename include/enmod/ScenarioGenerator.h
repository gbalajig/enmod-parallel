#ifndef ENMOD_SCENARIO_GENERATOR_H
#define ENMOD_SCENARIO_GENERATOR_H

#include "json.hpp"
#include <string>

using json = nlohmann::json;

class ScenarioGenerator {
public:
    static json generate(int size, const std::string& name);
};

#endif // ENMOD_SCENARIO_GENERATOR_H

