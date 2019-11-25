#include "config_file.h"
#include "cpptoml.h"
#include "utils/logger.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

using namespace Aperture;

///  Function to convert a string to a bool variable.
///
///  @param str   The string to be converted.
///  \return      The bool corresponding to str.
bool
to_bool(std::string& str) {
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  std::istringstream is(str);
  bool b;
  is >> std::boolalpha >> b;
  return b;
}

///  Whether we should skip the line in configuration file
bool
skip_line_or_word(const std::string& str) {
  if (str == "" || str[0] == '#')
    return true;
  else
    return false;
}

ConfigFile::ConfigFile() {
  Logger::print_info("Size of params is {}", sizeof(SimParams));
  // m_grid_conf.resize(3);
}

void
ConfigFile::parse_file(const std::string& filename, SimParams& params) {
  // std::cout << filename << std::endl;
  if (filename.empty()) throw(exceptions::file_not_found());

  auto config = cpptoml::parse_file(filename);
  SimParams defaults;

  params.delta_t =
      config->get_as<double>("delta_t").value_or(defaults.delta_t);
  params.max_ptc_number = config->get_as<uint64_t>("max_ptc_number")
                              .value_or(defaults.max_ptc_number);
  params.max_photon_number =
      config->get_as<uint64_t>("max_photon_number")
          .value_or(defaults.max_photon_number);
  params.ion_mass =
      config->get_as<double>("ion_mass").value_or(defaults.ion_mass);
  params.q_e = config->get_as<double>("q_e").value_or(defaults.q_e);
  params.num_species =
      config->get_as<int>("num_species").value_or(defaults.num_species);
  params.create_pairs = config->get_as<bool>("create_pairs")
                            .value_or(defaults.create_pairs);
  params.trace_photons = config->get_as<bool>("trace_photons")
                             .value_or(defaults.trace_photons);
  params.inject_ions =
      config->get_as<bool>("inject_ions").value_or(defaults.inject_ions);
  // params.use_bg_fields = config->get_as<bool>("use_bg_fields")
  //                            .value_or(defaults.use_bg_fields);
  params.gravity_on =
      config->get_as<bool>("gravity_on").value_or(defaults.gravity_on);
  params.gravity =
      config->get_as<double>("gravity").value_or(defaults.gravity);
  params.compactness = config->get_as<double>("compactness")
                           .value_or(defaults.compactness);
  params.rad_cooling_on = config->get_as<bool>("rad_cooling_on")
                              .value_or(defaults.rad_cooling_on);
  params.rad_cooling_coef = config->get_as<double>("rad_cooling_coef")
                                .value_or(defaults.rad_cooling_coef);

  params.track_percent = config->get_as<double>("track_percent")
                             .value_or(defaults.track_percent);
  params.gamma_thr =
      config->get_as<double>("gamma_thr").value_or(defaults.gamma_thr);
  params.spectral_alpha = config->get_as<double>("spectral_alpha")
                              .value_or(defaults.spectral_alpha);
  params.e_s = config->get_as<double>("e_s").value_or(defaults.e_s);
  params.e_min =
      config->get_as<double>("e_min").value_or(defaults.e_min);
  params.photon_path = config->get_as<double>("photon_path")
                           .value_or(defaults.photon_path);
  params.ic_path =
      config->get_as<double>("ic_path").value_or(defaults.ic_path);
  params.rad_energy_bins = config->get_as<int>("rad_energy_bins")
                               .value_or(defaults.rad_energy_bins);
  params.data_dir = config->get_as<std::string>("data_dir")
                        .value_or(defaults.data_dir);
  params.E_cutoff =
      config->get_as<double>("E_cutoff").value_or(defaults.E_cutoff);
  params.E_ph = config->get_as<double>("E_ph").value_or(defaults.E_ph);
  params.E_ph_min =
      config->get_as<double>("E_ph_min").value_or(defaults.E_ph_min);
  params.constE =
      config->get_as<double>("constE").value_or(defaults.constE);
  params.B0 = config->get_as<double>("B0").value_or(defaults.B0);
  params.BQ = config->get_as<double>("BQ").value_or(defaults.BQ);
  params.res_drag_coef = config->get_as<double>("res_drag_coef")
                             .value_or(defaults.res_drag_coef);
  params.star_kT =
      config->get_as<double>("star_kT").value_or(defaults.star_kT);
  params.damping_coef = config->get_as<double>("damping_coef")
                            .value_or(defaults.damping_coef);
  params.E_secondary = config->get_as<double>("E_secondary")
                           .value_or(defaults.E_secondary);
  params.r_cutoff =
      config->get_as<double>("r_cutoff").value_or(defaults.r_cutoff);
  params.omega =
      config->get_as<double>("omega").value_or(defaults.omega);
  params.a =
      config->get_as<double>("a").value_or(defaults.a);
  params.damping_length = config->get_as<uint64_t>("damping_length")
      .value_or(defaults.damping_length);

  params.l_curv = config->get_as<double>("l_curv").value_or(defaults.l_curv);
  params.e_curv = config->get_as<double>("e_curv").value_or(defaults.e_curv);
  params.l_ph = config->get_as<double>("l_ph").value_or(defaults.l_ph);

  auto periodic_boundary =
      config->get_array_of<bool>("periodic_boundary");
  if (periodic_boundary) {
    int n = periodic_boundary->size();
    params.periodic_boundary[0] = (*periodic_boundary)[0];
    if (n > 1) params.periodic_boundary[1] = (*periodic_boundary)[1];
    if (n > 2) params.periodic_boundary[2] = (*periodic_boundary)[2];
  }

  // Parse grid information
  auto mesh_table = config->get_table("Grid");
  if (mesh_table) {
    auto guard = mesh_table->get_array_of<int64_t>("guard");
    if (guard) {
      for (int i = 0; i < 3; i++) params.guard[i] = (*guard)[i];
    }

    auto N = mesh_table->get_array_of<int64_t>("N");
    if (N) {
      for (int i = 0; i < 3; i++) params.N[i] = (*N)[i];
    }

    auto lower = mesh_table->get_array_of<double>("lower");
    if (lower) {
      for (int i = 0; i < 3; i++) params.lower[i] = (*lower)[i];
    }

    auto size = mesh_table->get_array_of<double>("size");
    if (size) {
      for (int i = 0; i < 3; i++) params.size[i] = (*size)[i];
    }

    auto tile_size = mesh_table->get_array_of<double>("tile_size");
    if (tile_size) {
      for (int i = 0; i < 3; i++) params.tile_size[i] = (*tile_size)[i];
    }

    auto nodes = mesh_table->get_array_of<int64_t>("nodes");
    if (nodes) {
      for (int i = 0; i < 3; i++) params.nodes[i] = (*nodes)[i];
    }
  }

  // Simulation configuration
  auto sim_table = config->get_table("Simulation");
  if (sim_table) {
    params.coord_system = sim_table->get_as<std::string>("coord_system")
                              .value_or(defaults.coord_system);
    params.algorithm_ptc_move =
        sim_table->get_as<std::string>("algorithm_ptc_move")
            .value_or(defaults.algorithm_ptc_move);
    params.random_seed = sim_table->get_as<int>("random_seed")
                             .value_or(defaults.random_seed);
    params.downsample = sim_table->get_as<int>("downsample")
                            .value_or(defaults.downsample);
    params.max_steps = sim_table->get_as<int64_t>("max_steps")
                           .value_or(defaults.max_steps);
    params.data_interval = sim_table->get_as<int>("data_interval")
                               .value_or(defaults.data_interval);
    params.snapshot_interval =
        sim_table->get_as<int>("snapshot_interval")
            .value_or(defaults.snapshot_interval);
    params.sort_interval = sim_table->get_as<int>("sort_interval")
                               .value_or(defaults.sort_interval);
    params.current_smoothing =
        sim_table->get_as<int>("current_smoothing")
        .value_or(defaults.current_smoothing);
    params.update_fields =
        sim_table->get_as<bool>("update_fields")
        .value_or(defaults.update_fields);
    params.inject_particles =
        sim_table->get_as<bool>("inject_particles")
        .value_or(defaults.inject_particles);
  }

  // Radiation transfer parameters
  params.n_gamma =
      config->get_as<int64_t>("n_gamma").value_or(defaults.n_gamma);
  params.n_ep = config->get_as<int64_t>("n_ep").value_or(defaults.n_ep);

  compute_derived_quantities(params);
}

void
ConfigFile::compute_derived_quantities(SimParams& params) {
  params.log_file = params.data_dir + "output.log";
}
