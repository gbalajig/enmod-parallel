    #ifndef ENMOD_GRID_H
    #define ENMOD_GRID_H
    
    #include "Cost.h"
    #include "Types.h" 
    #include "json.hpp"
    #include <vector>
    #include <string>
    #include <map>
    #include "SensorTypes.h"
    
    using json = nlohmann::json;
    
    class Policy;
    
    struct FireEvent {
        Position pos;
        std::string size;
        int radius;
    };
    
    class Grid {
    public:
        Grid(const json& config);
        int getRows() const;
        int getCols() const;
        const std::string& getName() const;
        Position getStartPosition() const;
        const std::vector<Position>& getExitPositions() const;
        bool isValid(int r, int c) const;
        bool isWalkable(int r, int c) const;
        bool isExit(int r, int c) const;
        Cost getMoveCost(const Position& pos) const;
        Position getNextPosition(const Position& current, Direction dir) const;
        const json& getConfig() const;
        void addHazard(const json& event_config);
        CellType getCellType(const Position& pos) const;
        std::string getSmokeIntensity(const Position& pos) const;
    
        std::string toHtmlString() const;
        std::string toHtmlStringWithCost(const std::vector<std::vector<Cost>>& cost_map) const;
        std::string toHtmlStringWithPath(const std::vector<Position>& path) const;
        std::string toHtmlStringWithPolicy(const Policy& policy) const;
        std::string toHtmlStringWithAgent(const Position& agent_pos) const;
        void setCellUnwalkable(const Position& pos);
        void updateFromSensors(const std::vector<SensorReading>& readings);
    
    private:
        std::string grid_name;
        int rows;
        int cols;
        std::vector<std::vector<CellType>> grid_map;
        Position start_pos;
        std::vector<Position> exit_pos;
        json grid_config;
        std::map<Position, std::string> smoke_intensities;
        std::vector<FireEvent> active_fires;
    
        std::string cellToHtml(int r, int c, const std::string& content = "") const;
    };
    
    #endif // ENMOD_GRID_H
    

