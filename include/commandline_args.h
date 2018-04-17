#ifndef _COMMANDLINE_ARGS_H_
#define _COMMANDLINE_ARGS_H_

#include <string>
#include <cstdint>
#include <stdexcept>
#include <memory>
// #include "cxxopts.hpp"
// #include "boost/program_options.hpp"

namespace cxxopts {
class Options;
}

namespace Aperture {

namespace exceptions {

class program_option_terminate : public std::exception {
  virtual const char *what() const throw() {
    return "Program options finished.";
  }
};

}

class CommandArgs
{
 public:
  CommandArgs();
  ~CommandArgs();

  void read_args(int argc, char* argv[]);

  int dimx() const { return m_dimx; }
  int dimy() const { return m_dimy; }
  int dimz() const { return m_dimz; }
  uint32_t steps() const { return m_steps; }
  uint32_t data_interval() const { return m_data_interval; }
  const std::string& conf_filename() const { return m_conf_filename; }

 private:
  // default values provided in the constructor
  int m_dimx = 1, m_dimy = 1, m_dimz = 1;
  uint32_t m_steps, m_data_interval;
  std::string m_conf_filename;
  std::unique_ptr<cxxopts::Options> m_options;
  // boost::program_options::options_description _desc;
}; // ----- end of class commandline_args -----

}

#endif  // _COMMANDLINE_ARGS_H_
