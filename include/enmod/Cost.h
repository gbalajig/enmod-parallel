    #ifndef ENMOD_COST_H
    #define ENMOD_COST_H
    
    #include "Types.h"
    #include <iostream>
    #include <limits>
    
    const int MAX_COST = std::numeric_limits<int>::max();
    
    struct Cost {
        int smoke = MAX_COST;
        int time = MAX_COST;
        int distance = MAX_COST;
    
        inline static EvacuationMode current_mode = EvacuationMode::NORMAL;
    
        bool operator<(const Cost& other) const;
        bool operator>(const Cost& other) const;
        bool operator==(const Cost& other) const;
        Cost operator+(const Cost& other) const;
    };
    
    std::ostream& operator<<(std::ostream& os, const Cost& cost);
    
    #endif // ENMOD_COST_H
    

