#ifndef ENMOD_CUDA_TYPES_H
#define ENMOD_CUDA_TYPES_H

#include <limits>

#define DEVICE_MAX_INT 2147483647

enum class DeviceEvacuationMode { NORMAL, ALERT, PANIC };

struct DeviceCost {
    int smoke;
    int time;
    int distance;
};

struct DeviceGridCell {
    bool is_walkable;
    bool is_exit;
    DeviceCost move_cost; 
};

#endif // ENMOD_CUDA_TYPES_H