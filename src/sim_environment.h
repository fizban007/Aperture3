#ifndef _SIM_ENVIRONMENT_H_
#define _SIM_ENVIRONMENT_H_

#include "commandline_args.h"
#include "config_file.h"
#include "sim_params.h"

#include "core/multi_array.h"
#include "core/domain_info.h"
#include "core/grid.h"
#include "core/array.h"
#include "core/particles.h"
#include "core/photons.h"
#include <array>
#include <memory>
#include <random>
#include <string>
// #include "utils/mpi_comm.h"
#include <mpi.h>

namespace Aperture {

///  Class of the simulation environment. This class holds the basic
///  information that is useful for many other modules.
class sim_environment {
 public:
  sim_environment(int* argc, char*** argv);
  sim_environment(const std::string& conf_file);
  ~sim_environment();

  // Remove copy and assignment operators
  sim_environment(sim_environment const&) = delete;
  sim_environment& operator=(sim_environment const&) = delete;

  void setup_domain(int num_nodes);
  void setup_domain(int dimx, int dimy, int dimz = 1);
  // void setup_local_grid(Grid& local_grid, const Grid& super_grid,
  //                       const domain_info& info);
  void setup_local_grid();

  void send_array_guard_cells(multi_array<Scalar>& array);
  void send_array_guard_cells_single_dir(multi_array<Scalar>& array, int dim, int dir);

  void send_add_array_guard_cells(multi_array<Scalar>& array);
  void send_add_array_guard_cells_single_dir(multi_array<Scalar>& array, int dim, int dir);

  /// generate a random number between 0 and 1, useful for setting up
  /// things
  float gen_rand() { return m_dist(m_generator); }

  // data access methods
  const CommandArgs& args() const { return m_args; }
  SimParams& params() { return m_params; }
  const SimParams& params() const { return m_params; }
  const Grid& grid() const { return *m_grid; }
  const Grid& super_grid() const { return *m_super_grid; }
  const Grid& local_grid() const { return *m_grid; }
  Grid& local_grid() { return *m_grid; }
  const Quadmesh& mesh() const { return m_grid->mesh(); }

  const float* charges() const { return m_charges.data(); }
  const float* masses() const { return m_masses.data(); }
  const float* q_over_m() const { return m_q_over_m.data(); }
  float charge(int sp) const { return m_charges[sp]; }
  float mass(int sp) const { return m_masses[sp]; }
  float q_over_m(int sp) const { return m_q_over_m[sp]; }

  MPI_Comm world() const { return m_world; }
  const domain_info_t& domain_info() const { return m_domain_info; }
  // const MPICommWorld& world() const { return m_comm->world(); }
  // const MPICommCartesian& cartesian() const { return
  // m_comm->cartesian(); } const DomainInfo& domain_info() const {
  // return m_domain_info; }

  // void save_snapshot(cu_sim_data& data);
  // void load_snapshot(cu_sim_data& data);
  void load_from_snapshot(const std::string& snapshot_file);
  bool is_boundary(int bdy) const {
    return m_domain_info.is_boundary[bdy];
  }
  bool is_boundary(BoundaryPos bdy) const {
    return m_domain_info.is_boundary[(int)bdy];
  }

 protected:
  // sim_environment() {}
  void setup_env();
  void setup_env_extra();

  CommandArgs m_args;
  SimParams m_params;
  ConfigFile m_conf_file;

  std::unique_ptr<Grid> m_grid;
  std::unique_ptr<Grid> m_super_grid;

  MPI_Comm m_world;
  MPI_Comm m_cart;
  MPI_Datatype m_scalar_type;
  domain_info_t m_domain_info;

  // std::unique_ptr<MPIComm> m_comm;
  // std::unique_ptr<DataExporter> m_exporter;
  // std::unique_ptr<InitialCondition> m_ic;
  std::default_random_engine m_generator;
  std::uniform_real_distribution<float> m_dist;
  std::array<float, 8> m_charges;
  std::array<float, 8> m_masses;
  std::array<float, 8> m_q_over_m;

  std::vector<multi_array<Scalar>> m_send_buffers;
  std::vector<multi_array<Scalar>> m_recv_buffers;
  std::vector<particles_t> m_ptc_send_buffers;
  std::vector<particles_t> m_ptc_recv_buffers;
  std::vector<photons_t> m_ph_send_buffers;
  std::vector<photons_t> m_ph_recv_buffers;
  int m_dev_id = 0;
};  // ----- end of class sim_environment -----

}  // namespace Aperture

#endif  // _SIM_ENVIRONMENT_H_
