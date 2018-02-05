#ifndef  _LOGGER_H_
#define  _LOGGER_H_

#include <assert.h>
#include <iostream>
#include <fstream>
#include <string>
#include <functional>

#if __cplusplus > 199711L       // C++11 supported
////////////////////////////////////////////////////////////////////////////////
///  Define Logger::print as a function with variadic templates, only
///  when C++11 is supported. The syntax is a bit better than the
///  macro version. One needs to be careful when to use which
///  version.
///
///  Usage:
///
///      Logger::print(2, "Debug Message:", "a is", a);
///
///  A space is automatically added between the terms. Note that one
///  does not need to include << operator in the brackets. Just pass
///  in objects that you can usually pass in to cout clause between
///  the << operators.
///
///  Depending on the variable isOutputToFile, Logger::print can output
///  either to the screen ( the default mode ), or to a file specified
///  by logfile_default. Note if logfile_default is not open, contents
///  will be printed on the screen anyways.
///
///  Logger::print can also take as its first parameter a user-specified
///  ofstream object, followed by the same signature of ordinary
///  Logger::print. This outputs to the custom file. Note if the ofstream
///  object is not open, an error message and the contents will be
///  printed on the screen.
////////////////////////////////////////////////////////////////////////////////
namespace Logger {
extern int gVerbosityLvl;

void setVerbosityLevel (int lvl);

extern bool isLogToFile;

void setLogMethod ( bool isFile );

extern std::ofstream logfile_default;

void setLogFile ( const std::string& file );

void closeFile ();

template <typename T>
void print_screen(int vlevel, const T& t) {
  // Don't print anything if current verbosity level is higher than
  // the level specified in this command
  if (vlevel > gVerbosityLvl) return;
  std::cout << t << std::endl;
}

template <typename First, typename... Rest>
void print_screen(int vlevel, const First& first, const Rest&... rest) {
  // Don't print anything if current verbosity level is higher than
  // the level specified in this command
  if (vlevel > gVerbosityLvl) return;
  std::cout << first << " ";
  print_screen(vlevel, rest...); // recursive call using pack expansion syntax
}

template <typename T>
void err(const T& t) {
  // Always output error regardless of verbosity level
  std::cerr << t << std::endl;
}

template <typename First, typename... Rest>
void err(const First& first, const Rest&... rest) {
  // Always output error regardless of verbosity level
  std::cerr << first << " ";
  err(rest...); // recursive call using pack expansion syntax
}

template <typename T>
void print_file( int vlevel, const T& t ) {
  // Don't print anything if current verbosity level is higher than
  // the level specified in this command
  if (vlevel > gVerbosityLvl) return;
  logfile_default << t << std::endl;
}

template <typename First, typename... Rest>
void print_file( int vlevel, const First& first, const Rest&... rest ) {
  // Don't print anything if current verbosity level is higher than
  // the level specified in this command
  if (vlevel > gVerbosityLvl) return;
  logfile_default << first << " ";
  print_file( vlevel, rest...); // recursive call using pack expansion syntax
}

// print selects where to redirect stdout according to the variable isOutputToFile
// and the whether logfile_default is open.
template <typename First, typename... Rest>
void print(int vlevel, const First& first, const Rest&... rest) {
  if ( !isLogToFile || !logfile_default.is_open() ) print_screen( vlevel, first, rest... );
  else print_file( vlevel, first, rest... );
}

// use a user-specified ofstream object for output.
template <typename T>
void print( const std::ofstream& logfile_custom, int vlevel, const T& t ) {
  // Don't print anything if current verbosity level is higher than
  // the level specified in this command
  if (vlevel > gVerbosityLvl) return;
  if ( logfile_custom.is_open() )
    logfile_custom << t << std::endl;
  else {
    std::cout << "Error in opening file! " << std::endl;
    std::cout << t << std::endl;
  }
}

template <typename First, typename... Rest>
void print( const std::ofstream& logfile_custom, int vlevel, const First& first, const Rest&... rest ) {
  // Don't print anything if current verbosity level is higher than
  // the level specified in this command
  if (vlevel > gVerbosityLvl) return;
  if ( logfile_custom.is_open() ) {
    logfile_custom << first << " ";
    print( logfile_custom, vlevel, rest...); // recursive call using pack expansion syntax
  } else {
    std::cout << "Error in opening file! " << std::endl;
    print_screen( vlevel, first, rest... );
  }
}

}
#endif // __cplusplus > 199711L

////////////////////////////////////////////////////////////////////////////////
///  Defines a micro to output messages only in debug mode. Use this
///  when there is no C++11 support.
///  
///  Usage:
///  
///      MY_LOG("Debug Message");
///
///  Note that anything inside the bracket is directly passed into an
///  std::cout statement, so one can use something like the following:
///
///      MY_LOG("Parameter value: a = " << a);
////////////////////////////////////////////////////////////////////////////////
#ifndef   NDEBUG
#define MY_LOG(name) do {                                    \
        std::cerr << name << std::endl;                      \
    } while (0)

#else
    #define MY_LOG(name)
#endif // NDEBUG

#endif   // ----- #ifndef _LOGGER_H_  ----- 

