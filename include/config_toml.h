#ifndef _CONFIG_TOML_H_
#define _CONFIG_TOML_H_

#include <string>
#include "sim_params.h"

namespace Aperture {

class ConfigFile
{
 public:
  ConfigFile();
  ConfigFile(const std::string& filename);
  virtual ~ConfigFile();

  const SimParams& data() const { return m_data; }

  void parse_file();
  void parse_file(const std::string& filename);

 private:
  void def_params();

  std::string m_filename;
  SimParams m_data;
}; // ----- end of class ConfigFile -----



}

#endif  // _CONFIG_TOML_H_
