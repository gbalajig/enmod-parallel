#ifndef ENMOD_I_SENSOR_NETWORK_H
#define ENMOD_I_SENSOR_NETWORK_H

#include "SensorTypes.h"
#include <vector>

class ISensorNetwork {
public:
    virtual ~ISensorNetwork() = default;

    // Main method to poll all sensors in the network
    virtual std::vector<SensorReading> getAllReadings(double current_time) = 0;
};

#endif // ENMOD_I_SENSOR_NETWORK_H