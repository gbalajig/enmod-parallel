#include "enmod/Cost.h"

bool Cost::operator<(const Cost& other) const {
    switch (current_mode) {
        case EvacuationMode::PANIC:
        case EvacuationMode::ALERT:
            if (smoke != other.smoke) return smoke < other.smoke;
            if (time != other.time) return time < other.time;
            return distance < other.distance;
        case EvacuationMode::NORMAL:
        default:
            if (time != other.time) return time < other.time;
            if (distance != other.distance) return distance < other.distance;
            return smoke < other.smoke;
    }
}

bool Cost::operator>(const Cost& other) const { return other < *this; }
bool Cost::operator==(const Cost& other) const { return smoke == other.smoke && time == other.time && distance == other.distance; }

Cost Cost::operator+(const Cost& other) const {
    if (distance == MAX_COST || other.distance == MAX_COST) return {};
    return {smoke + other.smoke, time + other.time, distance + other.distance};
}

std::ostream& operator<<(std::ostream& os, const Cost& cost) {
    if (cost.distance == MAX_COST) {
        os << "[S:INF, T:INF, D:INF]";
    } else {
        os << "[S:" << cost.smoke << ", T:" << cost.time << ", D:" << cost.distance << "]";
    }
    return os;
}

