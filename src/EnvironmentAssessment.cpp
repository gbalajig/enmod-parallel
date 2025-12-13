#include "enmod/EnvironmentAssessment.h"
#include "enmod/Logger.h" // Include if you keep the logging
#include <cmath>          // For std::abs
#include <string>         // For std::to_string if logging

// Helper to convert mode to string
std::string modeToString(EvacuationMode mode) {
    switch (mode) {
        case EvacuationMode::NORMAL: return "NORMAL";
        case EvacuationMode::ALERT: return "ALERT";
        case EvacuationMode::PANIC: return "PANIC";
        default: return "UNKNOWN";
    }
}

// Define the global assessment function
EvacuationMode assessThreatAndSetMode(const Position& current_pos, const Grid& current_grid) {
    const auto& config = current_grid.getConfig(); // Get config directly from grid
    const auto& events = config.value("dynamic_events", json::array());
    EvacuationMode calculated_mode = EvacuationMode::NORMAL;

    // --- Logging Start ---
    // EvacuationMode original_mode = calculated_mode; // Store original for comparison if needed later
    // Logger::log(LogLevel::INFO, "Assess Start: Agent(" + std::to_string(current_pos.row) + "," + std::to_string(current_pos.col) + ") - Initial Mode: NORMAL");
    // --- End Logging Start ---


    for (const auto& event : events) {
        if (event.value("type", "") == "fire") {
            Position fire_pos = {event.at("position").at("row"), event.at("position").at("col")};
            // Check if the fire is currently active on the grid
            if (current_grid.getCellType(fire_pos) == CellType::FIRE) {
                int radius = 1; // Default radius for fire size "small" or unspecified
                std::string size = event.value("size", "small"); // Get size from event config
                if(size == "medium") radius = 2;
                else if(size == "large") radius = 3;

                int dist = std::abs(current_pos.row - fire_pos.row) + std::abs(current_pos.col - fire_pos.col);

                // --- Logging Check ---
                // Logger::log(LogLevel::INFO, "  Checking Fire at (" + std::to_string(fire_pos.row) + "," + std::to_string(fire_pos.col) +
                //                             "), Dist: " + std::to_string(dist) + ", Radius: " + std::to_string(radius));
                // --- End Logging Check ---


                if (dist <= 1) {
                     // --- Logging Trigger ---
                     // Logger::log(LogLevel::WARN, "    -> PANIC triggered by fire distance <= 1");
                     // --- End Logging Trigger ---
                    calculated_mode = EvacuationMode::PANIC;
                    // --- Logging End ---
                    // Logger::log(LogLevel::WARN, "Assess End: Mode set to " + modeToString(calculated_mode));
                    // --- End Logging End ---
                    return calculated_mode; // PANIC overrides everything
                }
                if (dist <= radius) {
                     // --- Logging Trigger ---
                     // Logger::log(LogLevel::INFO, "    -> ALERT triggered by fire distance <= radius");
                     // --- End Logging Trigger ---
                    calculated_mode = EvacuationMode::ALERT;
                    // Don't return; continue checking other fires and smoke
                }
            }
        }
    }

    // Check for heavy smoke only if not already in PANIC
    if (calculated_mode != EvacuationMode::PANIC) {
        int dr[] = {-1, 1, 0, 0};
        int dc[] = {0, 0, -1, 1};
        for(int i = 0; i < 4; ++i) {
            Position neighbor = {current_pos.row + dr[i], current_pos.col + dc[i]};
            if(current_grid.getSmokeIntensity(neighbor) == "heavy"){
                 // --- Logging Check ---
                 // Logger::log(LogLevel::INFO, "  Checking Smoke at (" + std::to_string(neighbor.row) + "," + std::to_string(neighbor.col) + ")");
                 // --- End Logging Check ---
                 if (calculated_mode != EvacuationMode::PANIC) { // Safety check
                     // --- Logging Trigger ---
                     // Logger::log(LogLevel::INFO, "    -> ALERT triggered by heavy smoke");
                     // --- End Logging Trigger ---
                     calculated_mode = EvacuationMode::ALERT;
                 }
            }
        }
    }

    // --- Logging End ---
    // if (original_mode != calculated_mode) { // Compare with initial state if needed
    //     Logger::log(LogLevel::WARN, "Assess End: Mode changed to " + modeToString(calculated_mode));
    // } else {
    //     // Logger::log(LogLevel::INFO, "Assess End: Mode remained " + modeToString(calculated_mode));
    // }
    // --- End Logging End ---

    return calculated_mode;
}