#include "enmod/ScenarioGenerator.h"
#include "enmod/Types.h"
#include "enmod/json.hpp"
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <vector>
#include <set> // For keeping track of spread locations

using json = nlohmann::json;

// --- Helper function Definitions ---
auto createSpreadingFire = [](json& config, Position origin, int start_time, const std::string& size_str, int grid_size) {
    if (!(origin.row >= 0 && origin.row < grid_size && origin.col >= 0 && origin.col < grid_size)) return;

    // --- Check if origin is a wall ---
    bool is_wall = false;
    if (config.contains("walls")) {
        is_wall = std::any_of(config["walls"].begin(), config["walls"].end(),
            [&origin](const json& wall_j){ return wall_j.value("row", -1) == origin.row && wall_j.value("col", -1) == origin.col; });
    }
    if (is_wall) return; // Don't start fire in a wall

    // --- Initial Fire Event ---
    config["dynamic_events"].push_back({
        {"type", "fire"}, {"time_step", start_time},
        {"position", {{"row", origin.row}, {"col", origin.col}}},
        {"size", size_str} // Keep the original size for the initial ignition
    });

    // --- Scaled Fire Spreading Logic ---
    int base_spread_rate = 4; // Timesteps between spread steps
    // Scale max_spread based on grid size, e.g., 10% of grid size, minimum 1
    int max_spread = std::max(1, grid_size / 10);
    // Determine initial radius based on input size_str for cost calculation reference
    int initial_radius = 1;
    if(size_str == "medium") initial_radius = 2;
    if(size_str == "large") initial_radius = 3;

    std::set<Position> fire_locations; // Keep track of where fire has spread
    fire_locations.insert(origin);
    std::vector<Position> current_front = {origin}; // Cells that will spread fire next step

    for (int spread_step = 1; spread_step <= max_spread; ++spread_step) {
        int current_time = start_time + spread_step * base_spread_rate;
        std::vector<Position> next_front;

        for (const auto& current_fire_pos : current_front) {
            // Spread to adjacent cells (including diagonals for faster spread)
            int dr[] = {-1, -1, -1,  0,  0,  1,  1,  1};
            int dc[] = {-1,  0,  1, -1,  1, -1,  0,  1};

            for(int i = 0; i < 8; ++i) {
                Position p = {current_fire_pos.row + dr[i], current_fire_pos.col + dc[i]};

                // Check bounds and if already on fire
                if (p.row >= 0 && p.row < grid_size && p.col >= 0 && p.col < grid_size &&
                    fire_locations.find(p) == fire_locations.end()) {

                    // Check if the target cell is a wall
                    bool p_is_wall = false;
                    if (config.contains("walls")) {
                        p_is_wall = std::any_of(config["walls"].begin(), config["walls"].end(),
                            [&p](const json& wall_j){ return wall_j.value("row", -1) == p.row && wall_j.value("col", -1) == p.col; });
                    }

                    if (!p_is_wall) {
                        // Add new fire event - keep subsequent fires "small" for simplicity unless complex behavior is desired
                        config["dynamic_events"].push_back({
                            {"type", "fire"},
                            {"time_step", current_time},
                            {"position", {{"row", p.row}, {"col", p.col}}},
                            {"size", "small"} // Keep spread events small
                        });
                        fire_locations.insert(p);
                        next_front.push_back(p);
                    }
                }
            }
        }
        current_front = std::move(next_front); // Update the fire front
        if (current_front.empty()) break; // Stop if fire can't spread further
    }
};

auto createDynamicSmoke = [](json& config, Position origin, int start_time, const std::string& intensity = "light", int grid_size = 0) {
     if (!(origin.row >= 0 && origin.row < grid_size && origin.col >= 0 && origin.col < grid_size)) return;
    // --- Check if origin is a wall ---
    bool is_wall = false;
    if (config.contains("walls")) {
        is_wall = std::any_of(config["walls"].begin(), config["walls"].end(),
            [&origin](const json& wall_j){ return wall_j.value("row", -1) == origin.row && wall_j.value("col", -1) == origin.col; });
    }
    if (is_wall) return; // Don't start smoke in a wall


    config["dynamic_events"].push_back({
        {"type", "smoke"}, {"time_step", start_time},
        {"position", {{"row", origin.row}, {"col", origin.col}}},
        {"intensity", intensity}
    });

    // --- Scaled Smoke Spreading ---
    int spread_rate = 5; // Smoke might spread slightly slower than fire
    int max_spread = std::max(1, grid_size / 15); // Spread less aggressively than fire maybe?

    std::set<Position> smoke_locations;
    smoke_locations.insert(origin);
    std::vector<Position> current_front = {origin};

    for (int spread_step = 1; spread_step <= max_spread; ++spread_step) {
        int current_time = start_time + spread_step * spread_rate;
        std::vector<Position> next_front;
        std::string current_intensity = (spread_step <= max_spread / 2 && intensity == "heavy") ? "heavy" : "light"; // Fade intensity?

        for (const auto& current_smoke_pos : current_front) {
             // Spread orthgonally first, then diagonally maybe? simpler: just orthogonally
            int dr[] = {-1, 1, 0, 0};
            int dc[] = {0, 0, -1, 1};

            for(int i = 0; i < 4; ++i) { // Only 4 directions
                 Position p = {current_smoke_pos.row + dr[i], current_smoke_pos.col + dc[i]};

                 if (p.row >= 0 && p.row < grid_size && p.col >= 0 && p.col < grid_size &&
                     smoke_locations.find(p) == smoke_locations.end()) {

                    // Check if the target cell is a wall
                    bool p_is_wall = false;
                    if (config.contains("walls")) {
                        p_is_wall = std::any_of(config["walls"].begin(), config["walls"].end(),
                            [&p](const json& wall_j){ return wall_j.value("row", -1) == p.row && wall_j.value("col", -1) == p.col; });
                    }

                    if (!p_is_wall) {
                        config["dynamic_events"].push_back({
                            {"type", "smoke"},
                            {"time_step", current_time},
                            {"position", {{"row", p.row}, {"col", p.col}}},
                            {"intensity", current_intensity} // Use potentially fading intensity
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
};

auto createBlockedPath = [](json& config, Position pos, int time_step, int /*grid_size*/) {
     // Basic implementation - just block the single cell if not wall/start/exit
     if (config.contains("walls")) {
        bool is_wall = std::any_of(config["walls"].begin(), config["walls"].end(),
            [&pos](const json& wall_j){ return wall_j.value("row", -1) == pos.row && wall_j.value("col", -1) == pos.col; });
        if (is_wall) return;
     }
     bool is_start = (config["start"]["row"] == pos.row && config["start"]["col"] == pos.col);
     bool is_exit = false;
     if (config.contains("exits")) {
         is_exit = std::any_of(config["exits"].begin(), config["exits"].end(),
            [&pos](const json& exit_j){ return exit_j.value("row", -1) == pos.row && exit_j.value("col", -1) == pos.col; });
     }
     if (!is_start && !is_exit) {
        config["dynamic_events"].push_back({
            {"type", "path_block"}, {"time_step", time_step},
            {"position", {{"row", pos.row}, {"col", pos.col}}}
        });
     }
};
// --- End Helper Functions ---


json ScenarioGenerator::generate(int size, const std::string& name) {
    json config;
    config["name"] = name;
    config["rows"] = size;
    config["cols"] = size;

    std::mt19937 rng(static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::uniform_int_distribution<int> dist(0, size - 1);

    Position start_pos = {1, 1}; // Fixed start
    config["start"] = {{"row", start_pos.row}, {"col", start_pos.col}};

    // --- Define Exits (Single Exit logic remains the same) ---
    config["exits"] = json::array();
    Position primary_exit = {size - 2, size - 2};
    // ... (rest of exit placement logic remains the same) ...
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


    // --- Minimal Wall Placement (Avoiding panic area logic remains the same) ---
    config["walls"] = json::array();
    int num_walls = (size * size) / 20; // Keep wall density low
    std::vector<Position> placed_walls;
    int placed_count = 0;
    int max_tries = num_walls * 10;
    int tries = 0;

    // Place fire AT panic point
    Position panic_point = { (start_pos.row + primary_exit.row) / 2 , (start_pos.col + primary_exit.col) / 2 };
    Position fire_point = panic_point; // Fire appears directly where agent is expected

    while(placed_count < num_walls && tries < max_tries) {
        // ... (wall placement logic remains the same, avoiding near_panic_area) ...
        tries++;
        int r = dist(rng);
        int c = dist(rng);
        Position p = {r, c};

        if (!(p.row >= 0 && p.row < size && p.col >= 0 && p.col < size)) continue;

        bool is_start = (p == start_pos);
        bool is_exit = std::any_of(config["exits"].begin(), config["exits"].end(),
            [&p](const json& exit_j){ return exit_j.value("row", -1) == p.row && exit_j.value("col", -1) == p.col; });
        bool already_placed = std::find(placed_walls.begin(), placed_walls.end(), p) != placed_walls.end();

        // Avoid start, exits, and the immediate panic test area (including fire_point itself)
        bool near_panic_area = (std::abs(p.row - panic_point.row) <= 1 && std::abs(p.col - panic_point.col) <= 1); // Keep area around fire clear

        if (!is_start && !is_exit && !already_placed && !near_panic_area) {
            config["walls"].push_back({{"row", p.row}, {"col", p.col}});
            placed_walls.push_back(p);
            placed_count++;
        }
    }


    // --- No Initial Smoke ---
    config["smoke"] = json::array();

    // --- Dynamic Events ---
    config["dynamic_events"] = json::array();

    // Calculate time to reach the panic point
    int time_to_panic_point = std::max(1, (panic_point.row - start_pos.row) + (panic_point.col - start_pos.col));

    // Trigger fire slightly earlier
    int fire_trigger_time = std::max(1, time_to_panic_point - 1); // Trigger 1 step before estimated arrival

    // --- CREATE THE PANIC TRAP with adjusted parameters ---
    std::string fire_size = "medium"; // Or "large"
    createSpreadingFire(config, fire_point, fire_trigger_time, fire_size, size); // Calls the updated helper

    // --- Add other dynamic events using updated helpers ---
    // Place them further away or at different times relative to the main fire event
    Position random_smoke_pos = {(start_pos.row + size -1)/2, (start_pos.col+size-1)/2+1}; // Some other location
    if (random_smoke_pos.row >= 0 && random_smoke_pos.col >= 0 && random_smoke_pos != fire_point) {
        // Use the updated createDynamicSmoke with scaling
        createDynamicSmoke(config, random_smoke_pos, std::max(1, fire_trigger_time / 3), "heavy", size);
    }
    Position random_block_pos = {1, size - 2};
     if (random_block_pos.row >= 0 && random_block_pos.col >= 0 && random_block_pos != fire_point) {
        createBlockedPath(config, random_block_pos, fire_trigger_time + size / 2, size); // Delay block significantly
     }

    return config;
}