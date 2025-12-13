#include "enmod/Grid.h"
#include "enmod/Policy.h" 
#include <stdexcept>
#include <sstream>
#include <cmath>
#include "enmod/SensorTypes.h"

Grid::Grid(const json& config) : grid_config(config) {
    try {
        grid_name = config.at("name");
        rows = config.at("rows");
        cols = config.at("cols");
        grid_map.assign(rows, std::vector<CellType>(cols, CellType::EMPTY));
        for (const auto& wall_pos : config.at("walls")) {
            grid_map[wall_pos.at("row")][wall_pos.at("col")] = CellType::WALL;
        }
        for (const auto& smoke_cfg : config.at("smoke")) {
            Position p = {smoke_cfg.at("row"), smoke_cfg.at("col")};
            grid_map[p.row][p.col] = CellType::SMOKE;
            smoke_intensities[p] = smoke_cfg.value("intensity", "light");
        }
        for (const auto& exit_p : config.at("exits")) {
            Position p = {exit_p.at("row"), exit_p.at("col")};
            grid_map[p.row][p.col] = CellType::EXIT;
            exit_pos.push_back(p);
        }
        start_pos = {config.at("start").at("row"), config.at("start").at("col")};
        grid_map[start_pos.row][start_pos.col] = CellType::START;
    } catch (const json::exception& e) {
        throw std::runtime_error("Failed to parse grid config: " + std::string(e.what()));
    }
}
int Grid::getRows() const { return rows; }
int Grid::getCols() const { return cols; }
const std::string& Grid::getName() const { return grid_name; }
Position Grid::getStartPosition() const { return start_pos; }
const std::vector<Position>& Grid::getExitPositions() const { return exit_pos; }
const json& Grid::getConfig() const { return grid_config; }
bool Grid::isValid(int r, int c) const { return r >= 0 && r < rows && c >= 0 && c < cols; }
bool Grid::isWalkable(int r, int c) const { if (!isValid(r, c)) return false; return grid_map[r][c] != CellType::WALL; }
bool Grid::isExit(int r, int c) const { if (!isValid(r,c)) return false; return grid_map[r][c] == CellType::EXIT; }
CellType Grid::getCellType(const Position& pos) const { if (!isValid(pos.row, pos.col)) return CellType::WALL; return grid_map[pos.row][pos.col]; }
std::string Grid::getSmokeIntensity(const Position& pos) const { if (smoke_intensities.count(pos)) { return smoke_intensities.at(pos); } return ""; }

void Grid::addHazard(const json& event_config) {
    Position pos = {event_config.at("position").at("row"), event_config.at("position").at("col")};
    if (!isValid(pos.row, pos.col)) {
        return; // Ignore events outside the grid
    }

    std::string type = event_config.value("type", "");

    if (type == "fire") {
        // --- NEW: Check if the cell is already FIRE, if so, maybe update radius? Or just skip? ---
        // For now, let's just overwrite/ensure it's fire. Could add logic later
        // to handle increasing fire size/intensity if it hits an existing fire cell.
        if (grid_map[pos.row][pos.col] != CellType::WALL) { // Don't overwrite walls
             grid_map[pos.row][pos.col] = CellType::FIRE;
             smoke_intensities.erase(pos); // Fire removes smoke
             std::string size = event_config.value("size", "small");
             int radius = 1;
             if(size == "medium") radius = 2;
             if(size == "large") radius = 3;
             // Consider managing active_fires better (e.g., updating existing fire radius)
             active_fires.push_back({pos, size, radius});
        }
    }
    else if (type == "path_block") {
        if (grid_map[pos.row][pos.col] != CellType::START && grid_map[pos.row][pos.col] != CellType::EXIT) {
             setCellUnwalkable(pos); // Make the cell a wall (if not start/exit)
             smoke_intensities.erase(pos); // Blockage removes smoke
             // Potentially remove from active_fires if one exists there?
        }
    }
    // --- NEW: Handle dynamic smoke ---
    else if (type == "smoke") {
        // Only add smoke to walkable, non-exit/start cells that aren't already on fire
        if (grid_map[pos.row][pos.col] == CellType::EMPTY || grid_map[pos.row][pos.col] == CellType::SMOKE) {
            grid_map[pos.row][pos.col] = CellType::SMOKE;
            smoke_intensities[pos] = event_config.value("intensity", "light");
        }
    }
    // --- NEW: Optional: Handle smoke clearing ---
    // else if (type == "clear_smoke") {
    //     if (grid_map[pos.row][pos.col] == CellType::SMOKE) {
    //         grid_map[pos.row][pos.col] = CellType::EMPTY; // Change back to empty
    //         smoke_intensities.erase(pos); // Remove intensity info
    //     }
    // }
    // --- END NEW ---
}

Cost Grid::getMoveCost(const Position& pos) const {
    if (!isValid(pos.row, pos.col)) return {};
    for(const auto& fire : active_fires){
        int dist = std::abs(pos.row - fire.pos.row) + std::abs(pos.col - fire.pos.col);
        if(dist <= fire.radius){
            if(dist == 0) return {1000, 100, 1};
            if(dist == 1) return {50, 10, 1};
            if(dist == 2) return {25, 5, 1};
            if(dist == 3) return {10, 2, 1};
        }
    }
    if (grid_map[pos.row][pos.col] == CellType::SMOKE) {
        if (smoke_intensities.count(pos) && smoke_intensities.at(pos) == "heavy") return {25, 4, 1};
        return {5, 2, 1};
    }
    return {0, 1, 1};
}
Position Grid::getNextPosition(const Position& current, Direction dir) const {
    switch (dir) {
        case Direction::UP: return {current.row - 1, current.col};
        case Direction::DOWN: return {current.row + 1, current.col};
        case Direction::LEFT: return {current.row, current.col - 1};
        case Direction::RIGHT: return {current.row, current.col + 1};
        default: return current;
    }
}
std::string Grid::cellToHtml(int r, int c, const std::string& content) const {
    std::string class_name; std::string text = content;
    switch (grid_map[r][c]) {
        case CellType::WALL: class_name = "wall"; text = "W"; break;
        case CellType::START: class_name = "start"; text = "S"; break;
        case CellType::EXIT: class_name = "exit"; text = "E"; break;
        case CellType::SMOKE: class_name = "smoke"; text = "~"; break;
        case CellType::FIRE: class_name = "fire"; text = "F"; break;
        default: break;
    }
    if (content.find("path") != std::string::npos) { class_name += " path"; }
    return "<td class='grid-cell " + class_name + "'>" + text + "</td>";
}
std::string Grid::toHtmlString() const {
     std::stringstream ss;
    ss << "<table class='grid-table'><tbody>";
    for(int i = 0; i < rows; ++i){
        ss << "<tr>";
        for(int j = 0; j < cols; ++j){ ss << cellToHtml(i,j); }
        ss << "</tr>";
    }
    ss << "</tbody></table>";
    return ss.str();
}
std::string Grid::toHtmlStringWithCost(const std::vector<std::vector<Cost>>& cost_map) const {
    std::stringstream ss;
    ss << "<table class='grid-table'><tbody>";
    for(int r = 0; r < rows; ++r){
        ss << "<tr>";
        for(int c = 0; c < cols; ++c){
             std::stringstream content_ss;
            if(cost_map[r][c].distance != MAX_COST) {
                content_ss << "S:" << cost_map[r][c].smoke << "<br>T:" << cost_map[r][c].time << "<br>D:" << cost_map[r][c].distance;
            }
            ss << cellToHtml(r,c, content_ss.str());
        }
        ss << "</tr>";
    }
    ss << "</tbody></table>";
    return ss.str();
}
std::string Grid::toHtmlStringWithPath(const std::vector<Position>& path) const {
     std::stringstream ss;
    ss << "<table class='grid-table'><tbody>";
    for(int r = 0; r < rows; ++r){
        ss << "<tr>";
        for(int c = 0; c < cols; ++c){
            bool is_path = false;
            for(const auto& p : path){ if(p.row == r && p.col == c){ is_path = true; break; } }
            ss << cellToHtml(r,c, is_path ? "path" : "");
        }
        ss << "</tr>";
    }
    ss << "</tbody></table>";
    return ss.str();
}
std::string Grid::toHtmlStringWithPolicy(const Policy& policy) const {
    std::stringstream ss;
    ss << "<table class='grid-table'><tbody>";
    for(int r = 0; r < rows; ++r){
        ss << "<tr>";
        for(int c = 0; c < cols; ++c){
            char dir_char = ' ';
            switch(policy.getDirection({r,c})){
                case Direction::UP: dir_char = '^'; break;
                case Direction::DOWN: dir_char = 'v'; break;
                case Direction::LEFT: dir_char = '<'; break;
                case Direction::RIGHT: dir_char = '>'; break;
                case Direction::STAY: dir_char = 'O'; break;
                default: break;
            }
            ss << cellToHtml(r,c, std::string(1, dir_char));
        }
        ss << "</tr>";
    }
    ss << "</tbody></table>";
    return ss.str();
}
std::string Grid::toHtmlStringWithAgent(const Position& agent_pos) const {
    std::stringstream ss;
    ss << "<table class='grid-table'><tbody>";
    for(int r = 0; r < rows; ++r){
        ss << "<tr>";
        for(int c = 0; c < cols; ++c){
            std::string content = "";
            if (r == agent_pos.row && c == agent_pos.col) { content = "A"; }
            ss << cellToHtml(r,c, content);
        }
        ss << "</tr>";
    }
    ss << "</tbody></table>";
    return ss.str();
}
void Grid::setCellUnwalkable(const Position& pos) {
    if (isValid(pos.row, pos.col)) {
        // Ensure we don't block an exit
        if (grid_map[pos.row][pos.col] != CellType::EXIT) {
            grid_map[pos.row][pos.col] = CellType::WALL;
        }
    }
}
void Grid::updateFromSensors(const std::vector<SensorReading>& readings) {
    // Clear dynamic hazards from the previous state if needed, 
    // or assume sensors provide a snapshot of the current state.
    // For this simulation, we assume we are overlaying new data.

    for (const auto& reading : readings) {
        if (!isValid(reading.location.row, reading.location.col)) continue;

        switch (reading.type) {
            case SensorType::THERMAL_CAMERA:
                if (reading.data == "FIRE_DETECTED") {
                    grid_map[reading.location.row][reading.location.col] = CellType::FIRE;
                    // Simplified: Add to active fires list for cost calc if not present
                    bool exists = false;
                    for(auto& f : active_fires) if(f.pos.row == reading.location.row && f.pos.col == reading.location.col) exists = true;
                    if(!exists) active_fires.push_back({reading.location, "small", 1}); // Default size/radius if sensor doesn't specify
                }
                break;

            case SensorType::SMOKE_DETECTOR:
                grid_map[reading.location.row][reading.location.col] = CellType::SMOKE;
                smoke_intensities[reading.location] = reading.data; // "light" or "heavy"
                break;

            case SensorType::LIDAR:
                // Detected debris/wall
                grid_map[reading.location.row][reading.location.col] = CellType::WALL;
                break;
            
            case SensorType::AGENT_BEACON:
                // Agent locations usually handled by the Controller, but Grid could mark them occupied
                break;
        }
    }
}