#ifndef ENMOD_REAL_TIME_SENSOR_NETWORK_H
#define ENMOD_REAL_TIME_SENSOR_NETWORK_H

#include "ISensorNetwork.h"
#include <string>

class RealTimeSensorNetwork : public ISensorNetwork {
public:
    RealTimeSensorNetwork(const std::string& data_file_path);
    
    // Reads the external JSON file and parses it into SensorReadings
    std::vector<SensorReading> getAllReadings(double current_time) override;

private:
    std::string file_path;
};

#endif // ENMOD_REAL_TIME_SENSOR_NETWORK_H
