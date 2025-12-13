#include "enmod/RealTimeSensorNetwork.h"
#include "enmod/json.hpp"
#include "enmod/Logger.h"
#include <fstream>
#include <iostream>

using json = nlohmann::json;

RealTimeSensorNetwork::RealTimeSensorNetwork(const std::string& data_file_path) 
    : file_path(data_file_path) {}

std::vector<SensorReading> RealTimeSensorNetwork::getAllReadings(double current_time) {
    std::vector<SensorReading> readings;
    std::ifstream f(file_path);

    if (!f.is_open()) {
        // It's okay if file isn't ready yet, just return empty
        return readings;
    }

    try {
        json j;
        f >> j; // Parse JSON

        for (const auto& item : j) {
            SensorReading r;
            r.sensor_id = item.value("id", "unknown");
            
            std::string type_str = item.value("type", "");
            if (type_str == "SMOKE") r.type = SensorType::SMOKE_DETECTOR;
            else if (type_str == "THERMAL") r.type = SensorType::THERMAL_CAMERA;
            else if (type_str == "LIDAR") r.type = SensorType::LIDAR;
            else if (type_str == "AGENT") r.type = SensorType::AGENT_BEACON;
            else continue; // Skip unknown types

            r.location = {item.value("row", 0), item.value("col", 0)};
            r.value = item.value("value", 0.0);
            r.data = item.value("data", "");
            r.timestamp = item.value("timestamp", 0.0);

            readings.push_back(r);
        }
    } catch (const json::exception& e) {
        Logger::log(LogLevel::WARN, "Failed to parse live sensor data: " + std::string(e.what()));
    }

    return readings;
}