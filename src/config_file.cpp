#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <fstream>
#include <sstream>
#include "config_file.h"
#include "utils/logger.h"

using namespace Aperture;

///  Function to convert a string to a bool variable.
///
///  @param str   The string to be converted.
///  \return      The bool corresponding to str.
bool to_bool(std::string& str) {
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  std::istringstream is(str);
  bool b;
  is >> std::boolalpha >> b;
  return b;
}

///  Whether we should skip the line in configuration file
bool skip_line_or_word(const std::string& str) {
  if (str == "" || str[0] == '#')
    return true;
  else
    return false;
}

// template <typename T>
// void add_param (json& data, const std::string& name, const T& value) {
//   if (data.find(name) != data.end()) return;
//   data[name] = value;
// }

ConfigFile::ConfigFile() {
  // m_grid_conf.resize(3);
}

ConfigFile::ConfigFile(const std::string& filename) {
  // m_grid_conf.resize(3);

  // Parse the configuration file
  try {
    parse_file(filename);
  } catch (const exceptions::empty_entry& e) {
    Logger::print_err("Error: {}, skipping.\n", e.what());
  } catch (const exceptions::file_not_found& e) {
    Logger::print_err("Error: {}\n", e.what());
  }
}

void
ConfigFile::parse_file(const std::string& filename) {
  if (filename.empty())
    throw (exceptions::file_not_found());

  m_filename = filename;
  std::string line;
  std::string word, input;
  std::ifstream file(filename.c_str());

  if (file) {
    while (std::getline(file, line)) {
      if (skip_line_or_word(line)) continue;

      std::stringstream parseline(line);
      parseline >> word >> std::ws;
      input = parseline.str().substr(parseline.tellg());
      if (input.empty())
        throw (exceptions::empty_entry(word));

      // transform to lower case to avoid case problems
      std::transform(word.begin(), word.end(), word.begin(), ::tolower);

      if (word.compare("metric") == 0) {
        m_data.metric = input;
      } else if (word.compare("delta_t") == 0) {
        m_data.delta_t = std::atof(input.c_str());
      } else if (word.compare("n_p") == 0) {
        m_data.ptc_per_cell = std::atoi(input.c_str());
      } else if (word.compare("q_e") == 0) {
        m_data.q_e = std::atof(input.c_str());
      } else if (word.compare("dim1") == 0 || word.compare("dim_1") == 0) {
        m_data.grid_config[0] = line;
      } else if (word.compare("dim2") == 0 || word.compare("dim_2") == 0) {
        m_data.grid_config[1] = line;
      } else if (word.compare("dim3") == 0 || word.compare("dim_3") == 0) {
        m_data.grid_config[2] = line;
      } else if (word.compare("data_dim1") == 0 || word.compare("data_dim_1") == 0) {
        m_data.data_grid_config[0] = line;
      } else if (word.compare("data_dim2") == 0 || word.compare("data_dim_2") == 0) {
        m_data.data_grid_config[1] = line;
      } else if (word.compare("data_dim3") == 0 || word.compare("data_dim_3") == 0) {
        m_data.data_grid_config[2] = line;
      } else if (word.compare("datadir") == 0) {
        m_data.data_dir = input;
      } else if (word.compare("gravity") == 0) {
        m_data.gravity = std::atof(input.c_str());
      } else if (word.compare("ion_mass") == 0) {
        m_data.ion_mass = std::atof(input.c_str());
      } else if (word.compare("max_part_num") == 0) {
        m_data.max_ptc_number = std::atol(input.c_str());
      } else if (word.compare("max_photon_num") == 0) {
        m_data.max_photon_number = std::atol(input.c_str());
      } else if (word.compare("periodic_boundary_1") == 0) {
        m_data.boundary_periodic[0] = to_bool(input);
      } else if (word.compare("periodic_boundary_2") == 0) {
        m_data.boundary_periodic[1] = to_bool(input);
      } else if (word.compare("periodic_boundary_3") == 0) {
        m_data.boundary_periodic[2] = to_bool(input);
      } else if (word.compare("interpolation_order") == 0) {
        m_data.interpolation_order = std::atoi(input.c_str());
      } else if (word.compare("create_pairs") == 0) {
        m_data.create_pairs = to_bool(input);
      } else if (word.compare("trace_photons") == 0) {
        m_data.trace_photons = to_bool(input);
      } else if (word.compare("gamma_thr") == 0) {
        m_data.gamma_thr = std::atof(input.c_str());
      } else if (word.compare("photon_path") == 0) {
        m_data.photon_path = std::atof(input.c_str());
      } else if (word.compare("ic_path") == 0) {
        m_data.ic_path = std::atof(input.c_str());
      } else if (word.compare("track_percent") == 0) {
        m_data.track_percent = std::atof(input.c_str());
      } else if (word.compare("data_dir") == 0) {
        m_data.data_dir = input;
      } else if (word.compare("data_file_prefix") == 0) {
        m_data.data_file_prefix = input;
      } else if (word.compare("data_compress") == 0) {
        m_data.data_compress = to_bool(input);
      } else if (word.compare("algorithm_ptc_move") == 0) {
        m_data.algorithm_ptc_move = input;
      } else if (word.compare("algorithm_ptc_push") == 0) {
        m_data.algorithm_ptc_push = input;
      } else if (word.compare("algorithm_field_update") == 0) {
        m_data.algorithm_field_update = input;
      } else if (word.compare("algorithm_current_deposit") == 0) {
        m_data.algorithm_current_deposit = input;
      } else if (word.compare("spectral_alpha") == 0) {
        m_data.spectral_alpha = std::atof(input.c_str());
      } else if (word.compare("e_s") == 0) {
        m_data.e_s = std::atof(input.c_str());
      } else if (word.compare("e_min") == 0) {
        m_data.e_min = std::atof(input.c_str());
      // } else if (word.compare("initial_condition") == 0) {
      //   m_data.initial_condition = input;
      } else {
        Logger::print_err("Unrecognized entry: {}\n", word);
      }
    }

    // m_data.grid_config = m_grid_conf;
    // for (int i = 0; i < 3; i++) {
    //   if (m_grid_conf[i] != "") {
    //     m_data["grid_conf"][i] = m_grid_conf[i];
    //   }
    // }

  } else {
    throw (exceptions::file_not_found(filename));
  }

}

void
ConfigFile::compute_derived_quantities() {

}
