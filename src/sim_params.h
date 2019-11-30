#ifndef _SIM_PARAMS_H_
#define _SIM_PARAMS_H_

#include "core/enum_types.h"
#include "core/typedefs.h"
#include "core/quadmesh.h"
#include <array>
#include <string>

namespace Aperture {

/////////////////////////////////////////////////////////////////////////
///  This is the standard simulation parameters class. This class will
///  be maintained in the environment class and be passed around as
///  reference to determine how the simulation will unfold.
/////////////////////////////////////////////////////////////////////////
struct SimParamsBase {
  // physical parameters
  Scalar delta_t = 0.01;
  Scalar q_e = 1.0;
  int ptc_per_cell = 1;
  uint64_t max_ptc_number = 100000;
  uint64_t max_photon_number = 100000;
  uint64_t ptc_buffer_size = 100000;
  uint64_t ph_buffer_size = 100000;
  Scalar ion_mass = 1.0;
  int num_species = 3;
  bool use_bg_fields = true;

  bool gravity_on = false;
  Scalar gravity = 0.0;
  Scalar compactness = 0.5;
  bool rad_cooling_on = false;
  Scalar rad_cooling_coef = 1.0;

  // std::array<std::string, 6> boundary_conditions;
  bool periodic_boundary[3] = {false};

  // simulation parameters
  int interpolation_order = 1;
  int current_smoothing = 3;
  bool data_compress = true;
  LogLevel log_lvl = LogLevel::debug;
  size_t max_steps = 1000;
  size_t data_interval = 100;
  int damping_length = 30;
  float damping_coef = 0.002;

  bool create_pairs = false;
  bool trace_photons = false;
  float gamma_thr = 20.0;
  bool annih_on = false;
  int annih_thr = 1000;
  float annih_fraction = 0.01;
  float track_percent = 0.2;
  float constE = 1.0;
  float B0 = 1.0;
  float BQ = 1.0;
  float star_kT = 0.001;
  float res_drag_coef = 40.0;
  float omega = 0.2;
  float a = 0.0;
  bool inject_ions = true;

  // These parameters are for radiative transfer
  float E_cutoff = 10.0;
  float E_ph = 4.0;
  float E_ph_min = 5.0;
  float spectral_alpha = 2.0;  // Slope of the soft photon spectrum
  float e_s = 0.2;  // separation between two regimes of pair creation
  float e_min = 1.0e-3;  // minimum energy of the background photons
  float photon_path = 1.0;
  float ic_path = 1.0;
  int rad_energy_bins = 256;
  float lph_cutoff = 1.0e4;
  float E_thr = 15.0;
  float E_secondary = 4.0;
  float r_cutoff = 4.0;

  // Curvature radiation parameters
  float l_curv = 6.0;
  float e_curv = 6.4e-9;
  float l_ph = 0.05;

  // Inverse compton parameters
  int n_gamma = 600;
  int n_ep = 600;

  // Domain decomposition parameters
  int dim_x = 1;
  int dim_y = 1;
  int dim_z = 1;

  // Mesh parameters
  int N[3] = {1, 1, 1};
  int guard[3] = {0, 0, 0};
  float lower[3] = {0.0, 0.0, 0.0};
  float size[3] = {1.0, 1.0, 1.0};
  int tile_size[3] = {1, 1, 1};
  int grid_dim = 1;
  int nodes[3] = {1, 1, 1};
};

struct SimParams : public SimParamsBase {
  std::string coord_system = "Cartesian";

  std::string data_dir = "../Data/";
  std::string data_file_prefix = "output";
  std::string log_method = "stdout";
  std::string log_file = data_dir + "output.log";
  std::string conf_file = "config.toml";

  // std::array<std::string, 3> grid_config;
  // std::array<std::string, 3> data_grid_config;

  // std::string algorithm_ptc_move = "mapping";
  std::string algorithm_ptc_move = "beadonwire";
  std::string algorithm_ptc_push = "Vay";
  std::string algorithm_field_update = "default";
  std::string algorithm_current_deposit = "Esirkepov";
  std::string initial_condition = "empty";

  int random_seed = 4321;
  int downsample = 1;
  int sort_interval = 20;
  int snapshot_interval = 1000;
  bool is_restart = false;
  bool update_fields = true;
  bool inject_particles = true;
};

}  // namespace Aperture

#endif  // _SIM_PARAMS_H_
