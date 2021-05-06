#ifndef _CONFIG_FILE_H_
#define _CONFIG_FILE_H_

#include "sim_params.h"
#include <stdexcept>
#include <string>

namespace Aperture {

namespace exceptions {

class file_not_found : public std::exception {
  std::string m_message;

 public:
  virtual const char* what() const throw() { return m_message.c_str(); }

  file_not_found() : m_message("File not found") {}
  file_not_found(const std::string& filename)
      : m_message("File not found: " + filename) {}
};

class empty_entry : public std::exception {
  std::string m_message;

 public:
  virtual const char* what() const throw() { return m_message.c_str(); }

  empty_entry() : m_message("Empty entry in config file") {}
  empty_entry(const std::string& entry)
      : m_message("Empty entry " + entry + " in config file") {}
};

}  // namespace exceptions

class ConfigFile {
 public:
  ConfigFile();
  ~ConfigFile() {}

  void parse_file(const std::string& filename, SimParams& params);

 private:
  void def_params();
  void compute_derived_quantities(SimParams& params);
};  // ----- end of class config_file -----

}  // namespace Aperture

#endif  // _CONFIG_FILE_H_