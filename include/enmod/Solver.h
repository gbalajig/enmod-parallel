    #ifndef ENMOD_SOLVER_H
    #define ENMOD_SOLVER_H
    
    #include "Grid.h"
    #include <string>
    #include <vector>
    #include <fstream> 
    
    class Solver {
    public:
        Solver(const Grid& grid_ref, const std::string& name);
        virtual ~Solver() = default;
    
        virtual void run() = 0;
        virtual Cost getEvacuationCost() const = 0;
        virtual void generateReport(std::ofstream& report_file) const = 0;
    
        const std::string& getName() const;
    
    protected:
        const Grid& grid;
        std::string solver_name;
    };
    
    #endif // ENMOD_SOLVER_H
    

