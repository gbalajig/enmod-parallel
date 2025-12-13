    #ifndef ENMOD_DYNAMIC_SOLVER_H
    #define ENMOD_DYNAMIC_SOLVER_H
    
    #include "Solver.h"
    #include "Types.h"
    
    struct StepReport {
        int time_step;
        Grid grid_state;
        Position agent_pos;
        std::string action;
        Cost current_total_cost;
        EvacuationMode mode;
    };
    
    #endif // ENMOD_DYNAMIC_SOLVER_H
    

