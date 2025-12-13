#include "enmod/Policy.h"
#include "enmod/Types.h" // Includes the new types header for Position/Direction
#include <iostream>

Policy::Policy(int r, int c) : rows(r), cols(c) {
    policy_map.assign(rows, std::vector<Direction>(cols, Direction::NONE));
}

void Policy::setDirection(const Position& pos, Direction dir) {
    if (pos.row >= 0 && pos.row < rows && pos.col >= 0 && pos.col < cols) {
        policy_map[pos.row][pos.col] = dir;
    }
}

Direction Policy::getDirection(const Position& pos) const {
    if (pos.row >= 0 && pos.row < rows && pos.col >= 0 && pos.col < cols) {
        return policy_map[pos.row][pos.col];
    }
    return Direction::NONE;
}

