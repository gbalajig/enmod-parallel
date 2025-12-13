#ifndef ENMOD_SENSOR_TYPES_H
#define ENMOD_SENSOR_TYPES_H

#include "Types.h"
#include <string>
#include <variant>

enum class SensorType {
    SMOKE_DETECTOR,
    THERMAL_CAMERA, // Detects Fire
    LIDAR,          // Detects Obstacles/Debris
    AGENT_BEACON    // Detects Agent ID/Location
};

struct SensorReading {
    std::string sensor_id;
    SensorType type;
    Position location;
    double value;       // e.g., smoke density (0-100), temp (C)
    std::string data;   // e.g., Agent ID string, or specific status "HEAVY"
    double timestamp;
};

#endif // ENMOD_SENSOR_TYPES_H