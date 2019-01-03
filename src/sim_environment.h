#ifndef _SIM_ENVIRONMENT_H_
#define _SIM_ENVIRONMENT_H_

#include "commandline_args.h"
#include "config_file.h"
#include "sim_params.h"

#include <memory>
#include <random>
#include <string>
#include <array>
// #include "data/domain_info.h"
#include "data/grid.h"
#include "utils/hdf_exporter.h"
// #include "utils/mpi_comm.h"
#include "utils/logger.h"

namespace Aperture {

////////////////////////////////////////////////////////////////////////////////
///  Class of the simulation environment. This class holds the basic
///  information that is useful for many other modules.
////////////////////////////////////////////////////////////////////////////////
class sim_environment {
 public:
  sim_environment(int* argc, char*** argv);
  sim_environment(const std::string& conf_file);
  ~sim_environment();

  // Remove copy and assignment operators
  sim_environment(sim_environment const&) = delete;
  sim_environment& operator=(sim_environment const&) = delete;

  // void setup_domain(int num_nodes); void
  // setup_domain(int dimx, int dimy, int dimz = 1); void
  // setup_local_grid(Grid& local_grid, const Grid& super_grid,
  //                       const DomainInfo& info);

  // void init_bg_fields(SimData& data);

  float gen_rand() { return m_dist(m_generator); }

  // data access methods
  const CommandArgs& args() const { return m_args; }
  SimParams& params() { return m_params; }
  const SimParams& params() const { return m_params; }
  // const ConfigFile& conf_file() const { return m_conf_file; }
  const Grid& grid() const { return *m_grid; }
  const Grid& local_grid() const { return *m_grid; }
  const Quadmesh& mesh() const { return m_grid->mesh(); }

  const float* charges() const { return m_charges.data(); }
  const float* masses() const { return m_masses.data(); }
  const float* q_over_m() const { return m_q_over_m.data(); }
  float charge(int sp) const { return m_charges[sp]; }
  float masse(int sp) const { return m_masses[sp]; }
  float q_over_m(int sp) const { return m_q_over_m[sp]; }

  // const Grid& super_grid() const { return m_super_grid; }
  // MetricType metric_type() const { return m_metric_type; }

  // DataExporter& exporter() { return *m_exporter; }
  // const MPICommWorld& world() const { return m_comm->world(); }
  // const MPICommCartesian& cartesian() const { return
  // m_comm->cartesian(); } const DomainInfo& domain_info() const {
  // return m_domain_info; }
  // // const BoundaryConditions& boundary_conditions() const { return
  // m_bc; }
  // // const InitialCondition& initial_condition() const { return
  // *m_ic; }

  // void save_snapshot(SimData& data);
  // void load_snapshot(SimData& data);
  void load_from_snapshot(const std::string& snapshot_file);

 protected:
  // sim_environment() {}
  void setup_env();

  CommandArgs m_args;
  SimParams m_params;
  ConfigFile m_conf_file;

  std::unique_ptr<Grid> m_grid;
  // Grid m_local_grid, m_local_grid_dual;
  // Grid m_super_grid;
  // Grid m_data_grid, m_data_super_grid;
  // MetricType m_metric_type;

  // DomainInfo m_domain_info;
  // BoundaryConditions m_bc;

  // std::unique_ptr<MPIComm> m_comm;
  // std::unique_ptr<DataExporter> m_exporter;
  // std::unique_ptr<InitialCondition> m_ic;
  std::default_random_engine m_generator;
  std::uniform_real_distribution<float> m_dist;
  std::array<float, 8> m_charges;
  std::array<float, 8> m_masses;
  std::array<float, 8> m_q_over_m;

};  // ----- end of class sim_environment -----


}

#endif  // _SIM_ENVIRONMENT_H_
