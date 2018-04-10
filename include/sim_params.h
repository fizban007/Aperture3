#ifndef _SIM_PARAMS_H_
#define _SIM_PARAMS_H_

#include <string>
#include <array>
#include "visit_struct/visit_struct.hpp"
#include "data/enum_types.h"

namespace Aperture {

typedef std::array<std::string, 6> bdy_cond_t;
typedef std::array<bool, 3> bdy_per_t;
typedef std::array<std::string, 3> grid_conf_t;

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

// VISITABLE_STRUCT(Aperture::SimParams,
//                           (std::string, metric)
//                           (double, delta_t)
//                           (double, q_e)
//                           (int, ptc_per_cell)
//                           (unsigned long, max_ptc_number)
//                           (unsigned long, max_photon_number)
//                           (double, ion_mass)
//                           (bool, gravity_on)
//                           (double, gravity)
//                           (Aperture::bdy_cond_t, boundary_conditions)
//                           (Aperture::bdy_per_t, boundary_periodic)
//                           (int, interpolation_order)
//                           (int, current_smoothing)
//                           (std::string, data_dir)
//                           (std::string, data_file_prefix)
//                           (bool, data_compress)
//                           (std::string, log_method)
//                           (std::string, log_file)
//                           (Aperture::LogLevel, log_lvl)
//                           (bool, create_pairs)
//                           (bool, trace_photons)
//                           (float, gamma_thr)
//                           (float, photon_path)
//                           (float, ic_path)
//                           (bool, annih_on)
//                           (int, annih_thr)
//                           (float, annih_fraction)
//                           (float, track_percent)
//                           (float, spectral_alpha)
//                           (float, e_s)
//                           (float, e_min)
//                           (Aperture::grid_conf_t, grid_config)
//                           (Aperture::grid_conf_t, data_grid_config)
//                           (std::string, algorithm_ptc_move)
//                           (std::string, algorithm_ptc_push)
//                           (std::string, algorithm_field_update)
//                           (std::string, algorithm_current_deposit)
//                           (std::string, initial_condition));

// VISITABLE_STRUCT(Aperture::SimParams,
//                  metric,
//                  delta_t,
//                  q_e,
//                  ptc_per_cell,
//                  max_ptc_number,
//                  max_photon_number,
//                  ion_mass,
//                  gravity_on,
//                  gravity,
//                  // boundary_conditions,
//                  boundary_periodic,
//                  interpolation_order,
//                  current_smoothing,
//                  data_dir,
//                  data_file_prefix,
//                  data_compress,
//                  log_method,
//                  log_file,
//                  log_lvl,
//                  create_pairs,
//                  trace_photons,
//                  gamma_thr,
//                  photon_path,
//                  ic_path,
//                  annih_on,
//                  annih_thr,
//                  annih_fraction,
//                  track_percent,
//                  spectral_alpha,
//                  e_s,
//                  e_min,
//                  // grid_config,
//                  // data_grid_config,
//                  algorithm_ptc_move,
//                  algorithm_ptc_push,
//                  algorithm_field_update,
//                  algorithm_current_deposit,
//                  initial_condition);


#endif  // _SIM_PARAMS_H_
