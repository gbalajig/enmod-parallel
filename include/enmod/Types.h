#ifndef ENMOD_TYPES_H
#define ENMOD_TYPES_H

enum class CellType { EMPTY, WALL, START, EXIT, SMOKE, FIRE };

struct Position {
    int row, col;
    bool operator==(const Position& other) const { return row == other.row && col == other.col; }
    bool operator!=(const Position& other) const { return !(*this == other); }
    bool operator<(const Position& other) const {
        if (row != other.row) return row < other.row;
        return col < other.col;
    }
};

enum class Direction { UP, DOWN, LEFT, RIGHT, STAY, NONE };

enum class EvacuationMode { NORMAL, ALERT, PANIC };

#endif // ENMOD_TYPES_H