#ifndef _COMMANDLINE_ARGS_H_
#define _COMMANDLINE_ARGS_H_

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace cxxopts {
class Options;
}

namespace Aperture {

struct SimParams;

namespace exceptions {

class program_option_terminate : public std::exception {
  virtual const char* what() const throw() {
    return "Program options finished.";
  }
};

}  // namespace exceptions

class CommandArgs {
 public:
  CommandArgs();
  ~CommandArgs();

  void read_args(int argc, char* argv[], SimParams& params);

 private:
  std::unique_ptr<cxxopts::Options> m_options;
};  // ----- end of class commandline_args -----

}  // namespace Aperture

#endif  // _COMMANDLINE_ARGS_H_
