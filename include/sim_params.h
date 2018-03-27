#ifndef _SIM_PARAMS_H_
#define _SIM_PARAMS_H_

#include <string>
#include <array>
#include "data/enum_types.h"

namespace Aperture {

////////////////////////////////////////////////////////////////////////////////
///  This is the standard simulation parameters class. This class will be
///  maintained in the environment class and be passed around as reference to
///  determine how the simulation will unfold.
////////////////////////////////////////////////////////////////////////////////
struct SimParams {
  std::string metric = "Cartesian";

  // physical parameters
  double        delta_t           = 0.01;
  double        q_e               = 1.0;
  int           ptc_per_cell      = 1;
  unsigned long max_ptc_number    = 100;
  unsigned long max_photon_number = 100;
  double        ion_mass          = 1.0;

  bool          gravity_on        = false;
  double        gravity           = 0.0;

  std::array<std::string, 6> boundary_conditions;
  std::array<bool, 3>        boundary_periodic = {false, false, false};

  // simulation parameters
  int         interpolation_order = 1;
  int         current_smoothing   = 0;
  std::string data_dir            = "../Data/";
  std::string data_file_prefix    = "output";
  bool        data_compress       = true;
  std::string log_method          = "file";
  std::string log_file            = data_dir + "logs/output.log";
  LogLevel    log_lvl             = LogLevel::debug;

  bool        create_pairs        = false;
  bool        trace_photons       = false;
  float       gamma_thr           = 20.0;
  float       photon_path         = 1.0;
  float       ic_path             = 1.0;

  bool          annih_on          = false;
  int           annih_thr         = 1000;
  float         annih_fraction    = 0.01;
  float         track_percent     = 0.2;

  // These parameters are for radiative transfer
  float       spectral_alpha      = 2.0;  // Slope of the soft photon spectrum
  float       e_s                 = 0.2;  // separation between two regimes of pair creation
  float       e_min               = 1.0e-3;  // minimum energy of the background photons

  std::array<std::string, 3> grid_config;
  std::array<std::string, 3> data_grid_config;

  std::string algorithm_ptc_move = "mapping";
  std::string algorithm_ptc_push = "Vay";
  std::string algorithm_field_update = "integral";
  std::string algorithm_current_deposit = "Esirkepov";
  std::string initial_condition = "empty";
};



}

#endif  // _SIM_PARAMS_H_
