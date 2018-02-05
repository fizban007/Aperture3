#include "utils/Logger.h"

namespace Logger {

const int maxVerbosityLvl = 3;

std::ofstream logfile_default;

bool isLogToFile = false;

/// Global verbosity level for output.
#ifndef NDEBUG
// Default to maximum verbosity level when in debug mode
int gVerbosityLvl = maxVerbosityLvl;
#else
int gVerbosityLvl = 0;
#endif // NDEBUG

void setVerbosityLevel (int lvl) {
  if (lvl < 5)
    gVerbosityLvl = lvl;
}

void setLogMethod (bool isFile) {
  isLogToFile = isFile ;
}

void setLogFile (const std::string &file ) {
  print( 0, " Opening ", file, " for logging. " );
  logfile_default.open( file );
}

void closeFile() {
  if ( logfile_default.is_open() )
    logfile_default.close();

  if ( isLogToFile ) {
    print( 0, "Logging file closed." );
    isLogToFile = false;
  }
}

}
