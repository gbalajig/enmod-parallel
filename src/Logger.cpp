#include "enmod/Logger.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

#ifdef _MSC_VER
#pragma warning(disable : 4996) // Disable warning for std::localtime
#endif

std::ofstream Logger::log_file;

void Logger::init(const std::string& filename) {
    log_file.open(filename, std::ios_base::out | std::ios_base::app);
    if (!log_file.is_open()) {
        std::cerr << "FATAL: Could not open log file: " << filename << std::endl;
    }
}

void Logger::log(LogLevel level, const std::string& message) {
    if (!log_file.is_open()) return;

    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");

    log_file << "[" << ss.str() << "] ";

    switch (level) {
        case LogLevel::INFO:  log_file << "[INFO] ";  break;
        case LogLevel::WARN:  log_file << "[WARN] ";  break;
        case LogLevel::ERROR: log_file << "[ERROR] "; break;
    }

    log_file << message << std::endl;
}

void Logger::close() {
    if (log_file.is_open()) {
        log_file.close();
    }
}

