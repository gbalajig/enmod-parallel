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
// (createSpreadingFire, createDynamicSmoke, createBlockedPathEvent helpers remain the same)
auto createSpreadingFire = [](json& config, Position origin, int start_time, const std::string& size_str, int grid_size) -> bool {
    if (!(origin.row >= 0 && origin.row < grid_size && origin.col >= 0 && origin.col < grid_size)) return false;
    bool is_wall = false;
    if (config.contains("walls")) {
        is_wall = std::any_of(config["walls"].begin(), config["walls"].end(),
            [&origin](const json& wall_j){ return wall_j.value("row", -1) == origin.row && wall_j.value("col", -1) == origin.col; });
    }
     bool is_start = (config["start"]["row"] == origin.row && config["start"]["col"] == origin.col);
    bool is_exit = false;
     if (config.contains("exits")) {
         is_exit = std::any_of(config["exits"].begin(), config["exits"].end(),
            [&origin](const json& exit_j){ return exit_j.value("row", -1) == origin.row && exit_j.value("col", -1) == origin.col; });
     }
    if (is_wall || is_start || is_exit) return false;

    config["dynamic_events"].push_back({
        {"type", "fire"}, {"time_step", start_time},
        {"position", {{"row", origin.row}, {"col", origin.col}}},
        {"size", size_str}
    });

    int base_spread_rate = 4;
    int max_spread = std::max(1, grid_size / 10);
    std::set<Position> fire_locations;
    fire_locations.insert(origin);
    std::vector<Position> current_front = {origin};

    for (int spread_step = 1; spread_step <= max_spread; ++spread_step) {
        int current_time = start_time + spread_step * base_spread_rate;
        std::vector<Position> next_front;
        for (const auto& current_fire_pos : current_front) {
            int dr[] = {-1, -1, -1,  0,  0,  1,  1,  1};
            int dc[] = {-1,  0,  1, -1,  1, -1,  0,  1};
            for(int i = 0; i < 8; ++i) {
                Position p = {current_fire_pos.row + dr[i], current_fire_pos.col + dc[i]};
                if (p.row >= 0 && p.row < grid_size && p.col >= 0 && p.col < grid_size &&
                    fire_locations.find(p) == fire_locations.end()) {
                    bool p_is_wall = false;
                    if (config.contains("walls")) {
                        p_is_wall = std::any_of(config["walls"].begin(), config["walls"].end(),
                            [&p](const json& wall_j){ return wall_j.value("row", -1) == p.row && wall_j.value("col", -1) == p.col; });
                    }
                     bool p_is_start = (config["start"]["row"] == p.row && config["start"]["col"] == p.col);
                     bool p_is_exit = false;
                     if (config.contains("exits")) {
                         p_is_exit = std::any_of(config["exits"].begin(), config["exits"].end(),
                            [&p](const json& exit_j){ return exit_j.value("row", -1) == p.row && exit_j.value("col", -1) == p.col; });
                     }
                    if (!p_is_wall && !p_is_start && !p_is_exit) {
                        config["dynamic_events"].push_back({
                            {"type", "fire"}, {"time_step", current_time},
                            {"position", {{"row", p.row}, {"col", p.col}}}, {"size", "small"}
                        });
                        fire_locations.insert(p);
                        next_front.push_back(p);
                    }
                }
            }
        }
        current_front = std::move(next_front);
        if (current_front.empty()) break;
    }
    return true; // Indicate success
};

auto createDynamicSmoke = [](json& config, Position origin, int start_time, const std::string& intensity = "light", int grid_size = 0) -> bool { // Return bool
     if (!(origin.row >= 0 && origin.row < grid_size && origin.col >= 0 && origin.col < grid_size)) return false;
    bool is_wall = false;
    if (config.contains("walls")) {
        is_wall = std::any_of(config["walls"].begin(), config["walls"].end(),
            [&origin](const json& wall_j){ return wall_j.value("row", -1) == origin.row && wall_j.value("col", -1) == origin.col; });
    }
    bool is_start = (config["start"]["row"] == origin.row && config["start"]["col"] == origin.col);
    bool is_exit = false;
     if (config.contains("exits")) {
         is_exit = std::any_of(config["exits"].begin(), config["exits"].end(),
            [&origin](const json& exit_j){ return exit_j.value("row", -1) == origin.row && exit_j.value("col", -1) == origin.col; });
     }
    if (is_wall || is_start || is_exit) return false;

    config["dynamic_events"].push_back({
        {"type", "smoke"}, {"time_step", start_time},
        {"position", {{"row", origin.row}, {"col", origin.col}}},
        {"intensity", intensity}
    });

    int spread_rate = 5;
    int max_spread = std::max(2, grid_size / 8); // Make smoke spread further
    std::set<Position> smoke_locations;
    smoke_locations.insert(origin);
    std::vector<Position> current_front = {origin};

    for (int spread_step = 1; spread_step <= max_spread; ++spread_step) {
        int current_time = start_time + spread_step * spread_rate;
        std::vector<Position> next_front;
        std::string current_intensity = (spread_step <= max_spread * 0.75 && intensity == "heavy") ? "heavy" : "light";
        for (const auto& current_smoke_pos : current_front) {
            int dr[] = {-1, 1, 0, 0}; // Orthogonal spread
            int dc[] = {0, 0, -1, 1};
            for(int i = 0; i < 4; ++i) {
                 Position p = {current_smoke_pos.row + dr[i], current_smoke_pos.col + dc[i]};
                 if (p.row >= 0 && p.row < grid_size && p.col >= 0 && p.col < grid_size &&
                     smoke_locations.find(p) == smoke_locations.end()) {
                    bool p_is_wall = false;
                    if (config.contains("walls")) {
                        p_is_wall = std::any_of(config["walls"].begin(), config["walls"].end(),
                            [&p](const json& wall_j){ return wall_j.value("row", -1) == p.row && wall_j.value("col", -1) == p.col; });
                    }
                     bool p_is_start = (config["start"]["row"] == p.row && config["start"]["col"] == p.col);
                     bool p_is_exit = false;
                     if (config.contains("exits")) {
                         p_is_exit = std::any_of(config["exits"].begin(), config["exits"].end(),
                            [&p](const json& exit_j){ return exit_j.value("row", -1) == p.row && exit_j.value("col", -1) == p.col; });
                     }
                    if (!p_is_wall && !p_is_start && !p_is_exit) {
                        config["dynamic_events"].push_back({
                            {"type", "smoke"}, {"time_step", current_time},
                            {"position", {{"row", p.row}, {"col", p.col}}}, {"intensity", current_intensity}
                        });
                        smoke_locations.insert(p);
                        next_front.push_back(p);
                    }
                 }
            }
        }
         current_front = std::move(next_front);
         if (current_front.empty()) break;
    }
     return true; // Indicate success
};

auto createBlockedPathEvent = [](json& config, Position pos, int time_step) -> bool {
     bool is_wall = false;
     if (config.contains("walls")) {
        is_wall = std::any_of(config["walls"].begin(), config["walls"].end(),
            [&pos](const json& wall_j){ return wall_j.value("row", -1) == pos.row && wall_j.value("col", -1) == pos.col; });
     }
      bool is_start = (config["start"]["row"] == pos.row && config["start"]["col"] == pos.col);
     bool is_exit = false;
     if (config.contains("exits")) {
         is_exit = std::any_of(config["exits"].begin(), config["exits"].end(),
            [&pos](const json& exit_j){ return exit_j.value("row", -1) == pos.row && exit_j.value("col", -1) == pos.col; });
     }
     if (!is_wall && !is_start && !is_exit) {
        config["dynamic_events"].push_back({
            {"type", "path_block"}, {"time_step", time_step},
            {"position", {{"row", pos.row}, {"col", pos.col}}}
        });
        return true; // Indicate success
     }
     return false; // Indicate failure
};
// --- End Helper Functions ---


json ScenarioGenerator::generate(int size, const std::string& name) {
    json config;
    config["name"] = name;
    config["rows"] = size;
    config["cols"] = size;

    std::mt19937 rng(static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::uniform_int_distribution<int> dist_pos(0, size - 1);
    
    Position start_pos = {1, 1};
    config["start"] = {{"row", start_pos.row}, {"col", start_pos.col}};

    // --- Define Exits (Single Exit) ---
    config["exits"] = json::array();
    Position primary_exit = {size - 2, size - 2};
     if (primary_exit.row >= 0 && primary_exit.col >= 0 && !(primary_exit == start_pos)) {
        config["exits"].push_back({{"row", primary_exit.row}, {"col", primary_exit.col}});
    } else if (size > 1) {
         primary_exit = {size - 1, size - 1};
         if (!(primary_exit == start_pos)) {
            config["exits"].push_back({{"row", primary_exit.row}, {"col", primary_exit.col}});
         }
    }
    if (config["exits"].empty()) {
         if (size > 1) {
             primary_exit = {size - 1, 0};
             config["exits"].push_back({{"row", primary_exit.row}, {"col", primary_exit.col}});
         } else {
             throw std::runtime_error("Cannot generate a valid scenario for grid size " + std::to_string(size));
         }
    }
    primary_exit = {config["exits"][0]["row"], config["exits"][0]["col"]};


    // --- REVERTED TO SIMPLE WALL PLACEMENT ---
    config["walls"] = json::array();
    int num_walls = (size * size) / 20; // Original low wall density
    std::set<Position> placed_locations; 
    placed_locations.insert(start_pos);
    placed_locations.insert(primary_exit);

    // --- Define key points for hazards ---
    // Hazard 1 (PANIC): Adjacent to start
    Position panic_fire_pos_1 = {std::max(1, size / 10), std::max(1, size / 10)}; // Relative to size
    if (panic_fire_pos_1 == start_pos) panic_fire_pos_1.row++;
    placed_locations.insert(panic_fire_pos_1); 

    // Hazard 2 (PANIC): Near Exit
    Position exit_trap_fire_pos = {primary_exit.row - (size / 10), primary_exit.col - (size / 10)}; // Relative to size
    if(exit_trap_fire_pos.row < 0) exit_trap_fire_pos.row = size / 2 + 2; // boundary check
    if (exit_trap_fire_pos == panic_fire_pos_1) exit_trap_fire_pos.row--; // offset
    placed_locations.insert(exit_trap_fire_pos);


    // --- *** NEW: VARIABLE MID-PATH HAZARDS *** ---
    int num_mid_hazards = size / 10; // Scale number of hazards with grid size
    // Place hazards in the "middle" 60% of the grid
    std::uniform_int_distribution<int> dist_pos_mid(size / 5, size - (size / 5)); 
    // Time hazards to appear throughout the journey
    int max_time = static_cast<int>(size * 1.5); // Estimated travel time
    std::uniform_int_distribution<int> dist_time(5, std::max(10, max_time)); 

    for (int i = 0; i < num_mid_hazards; ++i) {
        Position p;
        int placement_tries = 0;
        // Try to find an empty spot 50 times
        do {
            p = {dist_pos_mid(rng), dist_pos_mid(rng)};
            placement_tries++;
        } while (placed_locations.count(p) && placement_tries < 50);

        // If we found a free spot, reserve it
        if (!placed_locations.count(p)) {
            placed_locations.insert(p);
        }
    }
    // --- *** END VARIABLE HAZARDS *** ---


    // --- Place random walls, avoiding start, exit, and ALL hazard zones ---
    int placed_count = 0;
    int max_tries = (num_walls + size) * 5;
    int tries = 0;
    while(placed_count < num_walls && tries < max_tries) {
        tries++;
        int r = dist_pos(rng);
        int c = dist_pos(rng);
        Position p = {r, c};

        if (placed_locations.find(p) == placed_locations.end()) {
            config["walls"].push_back({{"row", p.row}, {"col", p.col}});
            placed_locations.insert(p);
            placed_count++;
        }
    }


    // --- No Initial Smoke ---
    config["smoke"] = json::array();

    // --- Dynamic Events ---
    config["dynamic_events"] = json::array();

    // --- ADD GUARANTEED TRAPS ---
    // 1. PANIC Test 1: Fire adjacent to start
    int fire_trigger_time_1 = 3; 
    createSpreadingFire(config, panic_fire_pos_1, fire_trigger_time_1, "large", size);

    // 2. PANIC Test 2 (Exit Trap)
    int fire_trigger_time_exit = std::max(15, max_time - (size / 10)); // Timed to catch the agent near the end
    createSpreadingFire(config, exit_trap_fire_pos, fire_trigger_time_exit, "large", size);
    
    // --- ADD VARIABLE HAZARDS ---
    // We re-iterate over the locations we reserved, skipping the ones we just used
    // (This is a bit redundant but ensures helpers are called correctly)
    std::set<Position> guaranteed_hazards = {panic_fire_pos_1, exit_trap_fire_pos};
    for (const auto& p : placed_locations) {
        if (p == start_pos || p == primary_exit || guaranteed_hazards.count(p)) {
            continue; // Skip start, exit, and guaranteed traps
        }
        if (config["walls"].dump().find("{\"col\":" + std::to_string(p.col) + ",\"row\":" + std::to_string(p.row) + "}") != std::string::npos) {
            continue; // Skip if it's a wall (shouldn't happen, but good check)
        }

        int time = dist_time(rng);
        int type = rng() % 3;

        if (type == 0) {
            createSpreadingFire(config, p, time, "large", size);
        } else if (type == 1) {
            createDynamicSmoke(config, p, time, "heavy", size);
        } else {
            createBlockedPathEvent(config, p, time);
        }
    }
    
    return config;
}