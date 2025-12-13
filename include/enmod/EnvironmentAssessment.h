#ifndef ENMOD_ENVIRONMENT_ASSESSMENT_H
#define ENMOD_ENVIRONMENT_ASSESSMENT_H

#include "enmod/Grid.h" // Needs Grid to check cell types and events
#include "enmod/Types.h" // Needs Position and EvacuationMode

// Declare the global assessment function
EvacuationMode assessThreatAndSetMode(const Position& current_pos, const Grid& current_grid);

// Optional: Helper to convert mode to string (useful for logging)
std::string modeToString(EvacuationMode mode);

#endif // ENMOD_ENVIRONMENT_ASSESSMENT_H