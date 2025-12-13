    #ifndef ENMOD_POLICY_H
    #define ENMOD_POLICY_H
    
    #include "Types.h"
    #include <vector>
    
    class Policy {
    public:
        Policy(int r, int c);
        void setDirection(const Position& pos, Direction dir);
        Direction getDirection(const Position& pos) const;
    
    private:
        int rows;
        int cols;
        std::vector<std::vector<Direction>> policy_map;
    };
    
    #endif // ENMOD_POLICY_H
    

