#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <fstream>
#include <sstream>
#include "config_file.h"
#include "utils/logger.h"
#include "cpptoml.h"

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

void
ConfigFile::parse_file(const std::string& filename, SimParams& params) {
  // std::cout << filename << std::endl;
  if (filename.empty())
    throw (exceptions::file_not_found());

  auto config = cpptoml::parse_file(filename);
  SimParams defaults;

  params.delta_t = config->get_as<double>("delta_t")
                   .value_or(defaults.delta_t);
  params.max_ptc_number = config->get_as<unsigned long>("max_ptc_number")
                          .value_or(defaults.max_ptc_number);
  params.max_photon_number = config->get_as<unsigned long>("max_photon_number")
                             .value_or(defaults.max_photon_number);
  params.ion_mass = config->get_as<double>("ion_mass")
                   .value_or(defaults.ion_mass);
  params.q_e = config->get_as<double>("q_e")
               .value_or(defaults.q_e);
  params.create_pairs = config->get_as<bool>("create_pairs")
                        .value_or(defaults.create_pairs);
  params.trace_photons = config->get_as<bool>("trace_photons")
                        .value_or(defaults.trace_photons);
  params.track_percent = config->get_as<double>("track_percent")
                         .value_or(defaults.track_percent);
  params.gamma_thr = config->get_as<double>("gamma_thr")
                         .value_or(defaults.gamma_thr);
  params.spectral_alpha = config->get_as<double>("spectral_alpha")
                         .value_or(defaults.spectral_alpha);
  params.e_s = config->get_as<double>("e_s")
                         .value_or(defaults.e_s);
  params.e_min = config->get_as<double>("e_min")
                 .value_or(defaults.e_min);
  params.photon_path = config->get_as<double>("photon_path")
                       .value_or(defaults.photon_path);
  params.ic_path = config->get_as<double>("ic_path")
                       .value_or(defaults.ic_path);
  params.data_dir = config->get_as<std::string>("data_dir")
                    .value_or(defaults.data_dir);
  auto periodic_boundary = config->get_array_of<bool>("periodic_boundary");
  if (periodic_boundary) {
    int n = periodic_boundary->size();
    params.periodic_boundary[0] = (*periodic_boundary)[0];
    if (n > 1) params.periodic_boundary[1] = (*periodic_boundary)[1];
    if (n > 2) params.periodic_boundary[2] = (*periodic_boundary)[2];
  }

  // Parse grid information
  auto mesh_table = config->get_table("Mesh");
  auto guard = mesh_table->get_array_of<int64_t>("guard");
  if (guard) { for (int i = 0; i < 3; i++) params.guard[i] = (*guard)[i]; }

  auto N = mesh_table->get_array_of<int64_t>("N");
  if (N) { for (int i = 0; i < 3; i++) params.N[i] = (*N)[i]; }

  auto lower = mesh_table->get_array_of<double>("lower");
  if (lower) { for (int i = 0; i < 3; i++) params.lower[i] = (*lower)[i]; }

  auto size = mesh_table->get_array_of<double>("size");
  if (size) { for (int i = 0; i < 3; i++) params.size[i] = (*size)[i]; }

  params.tile_size = mesh_table->get_as<int64_t>("tile_size")
                     .value_or(defaults.tile_size);

  compute_derived_quantities(params);
}

void
ConfigFile::compute_derived_quantities(SimParams& params) {
  params.log_file = params.data_dir + "output.log";
}
