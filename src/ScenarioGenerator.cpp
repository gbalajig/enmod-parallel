#include "enmod/ScenarioGenerator.h"
#include "enmod/Types.h"
#include "enmod/json.hpp"
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <vector>
#include <set>

using json = nlohmann::json;

// --- Helper function Definitions ---
auto createSpreadingFire = [](json& config, Position origin, int start_time, const std::string& size_str, int grid_size) -> bool {
    if (!(origin.row >= 0 && origin.row < grid_size && origin.col >= 0 && origin.col < grid_size)) return false;
    
    // Check overlapping
    if (config.contains("walls")) {
         for (const auto& w : config["walls"]) if(w.value("row",-1)==origin.row && w.value("col",-1)==origin.col) return false;
    }
    if (config["start"]["row"] == origin.row && config["start"]["col"] == origin.col) return false;
    if (config.contains("exits")) {
        for (const auto& e : config["exits"]) if(e.value("row",-1)==origin.row && e.value("col",-1)==origin.col) return false;
    }

    config["dynamic_events"].push_back({
        {"type", "fire"}, {"time_step", start_time},
        {"position", {{"row", origin.row}, {"col", origin.col}}},
        {"size", size_str}
    });
    
    // Simple spread logic
    int base_spread_rate = 4;
    int max_spread = std::max(1, grid_size / 2);
    
    return true; 
};

json ScenarioGenerator::generate(int size, const std::string& name, double obstacle_density) {
    json config;
    config["name"] = name;
    config["rows"] = size;
    config["cols"] = size;

    std::mt19937 rng(static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::uniform_int_distribution<int> dist_pos(0, size - 1);
    
    // 1. Setup Start (Top-Left) and Exit (Bottom-Right)
    Position start_pos = {0, 0}; 
    config["start"] = {{"row", start_pos.row}, {"col", start_pos.col}};

    config["exits"] = json::array();
    Position primary_exit = {size - 1, size - 1};
    config["exits"].push_back({{"row", primary_exit.row}, {"col", primary_exit.col}});

    // 2. Generate Static Obstacles based on Density
    config["walls"] = json::array();
    
    // Calculate exact number of walls needed
    int total_cells = size * size;
    int num_walls = static_cast<int>(total_cells * obstacle_density);
    
    std::set<Position> placed_locations; 
    placed_locations.insert(start_pos);
    placed_locations.insert(primary_exit);

    int placed_count = 0;
    int max_tries = num_walls * 10;
    int tries = 0;
    
    while(placed_count < num_walls && tries < max_tries) {
        tries++;
        int r = dist_pos(rng);
        int c = dist_pos(rng);
        Position p = {r, c};

        // If cell is empty, place a wall
        if (placed_locations.find(p) == placed_locations.end()) {
            config["walls"].push_back({{"row", p.row}, {"col", p.col}});
            placed_locations.insert(p);
            placed_count++;
        }
    }

    // 3. Initialize Smoke (Fixed missing key error)
    config["smoke"] = json::array();

    // 4. Add Dynamic Hazards (Fire)
    config["dynamic_events"] = json::array();
    
    // Place a fire near the center to force re-planning, but ensure it doesn't spawn ON a wall
    Position center = {size/2, size/2};
    int hazard_attempts = 0;
    while(placed_locations.find(center) != placed_locations.end() && hazard_attempts < 20) {
        center = {dist_pos(rng), dist_pos(rng)};
        hazard_attempts++;
    }

    if(placed_locations.find(center) == placed_locations.end()) {
         config["dynamic_events"].push_back({
            {"type", "fire"}, {"time_step", 5},
            {"position", {{"row", center.row}, {"col", center.col}}},
            {"size", "large"}
        });
    }

    return config;
}