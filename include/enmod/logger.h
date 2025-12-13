#ifndef ENMOD_LOGGER_H
#define ENMOD_LOGGER_H

#include <string>
#include <fstream>

enum class LogLevel { INFO, WARN, ERROR };

class Logger {
public:
    static void init(const std::string& filename);
    static void log(LogLevel level, const std::string& message);
    static void close();

private:
    static std::ofstream log_file;
};

#endif // ENMOD_LOGGER_H

