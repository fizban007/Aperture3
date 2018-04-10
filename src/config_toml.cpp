#include "config_toml.h"
#include "cpptoml.h"
#include "utils/logger.h"
#include "visit_struct/visit_struct.hpp"

namespace Aperture {

ConfigFile::ConfigFile() {}

ConfigFile::ConfigFile(const std::string& filename) :
    m_filename(filename) {
  parse_file();
}

ConfigFile::~ConfigFile() {}


void
ConfigFile::parse_file() {
  if (!m_filename.empty())
    parse_file(m_filename);
  else
    Logger::print_err("Empty config filename!");
}

void
ConfigFile::parse_file(const std::string &filename) {
  if (filename.empty())
    Logger::print_err("Empty config filename!");

  auto config = cpptoml::parse_file(filename);

  visit_struct::for_each(m_data, [config](const char * name, auto & value) {
      typedef typename std::remove_reference<decltype(value)>::type x_type;
      auto val = config->get_as<x_type>(name);
      if (val)
        value = *val;
    });
}


}
