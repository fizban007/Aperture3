#ifndef _SIM_ENVIRONMENT_DEV_H_
#define _SIM_ENVIRONMENT_DEV_H_

#include "commandline_args.h"
#include "config_file.h"
#include "sim_params.h"
#include <memory>
#include <random>
#include <string>
// #include "data/domain_info.h"
#include "data/grid.h"
#include "data/fields_dev.h"
// #include "utils/hdf_exporter.h"
// #include "utils/mpi_comm.h"
#include "utils/logger.h"
#include "sim_environment.h"

namespace Aperture {

struct cu_sim_data;
// class DomainCommunicator;

////////////////////////////////////////////////////////////////////////////////
///  Class of the simulation environment. This class holds the basic
///  information that is useful for many other modules.
////////////////////////////////////////////////////////////////////////////////
class Environment : public sim_environment {
 public:
  Environment(int* argc, char*** argv);
  Environment(const std::string& conf_file);
  ~Environment();

  // Remove copy and assignment operators
  Environment(Environment const&) = delete;
  Environment& operator=(Environment const&) = delete;

  // void set_initial_condition(cu_sim_data& data);
  // void set_initial_condition(cu_sim_data& data, const Index& start, const
  // Extent& extent); void setup_domain(int num_nodes); void
  // setup_domain(int dimx, int dimy, int dimz = 1); void
  // setup_local_grid(Grid& local_grid, const Grid& super_grid,
  //                       const DomainInfo& info);

  // void set_initial_condition(InitialCondition* ic);

  // void add_output(const std::string& name);
  void init_bg_fields(cu_sim_data& data);

  // float gen_rand() { return m_dist(m_generator); }

  // data access methods
  // const CommandArgs& args() const { return m_args; }
  // const SimParams& conf() const { return m_conf_file.data(); }
  // SimParams& params() { return m_params; }
  // const SimParams& params() const { return m_params; }
  // const ConfigFile& conf_file() const { return m_conf_file; }
  // const Grid& grid() const { return *m_grid; }
  // const Grid& local_grid() const { return *m_grid; }
  // const Quadmesh& mesh() const { return m_grid->mesh(); }
  // cu_vector_field<Scalar>& E_bg() { return m_Ebg; }
  // cu_vector_field<Scalar>& B_bg() { return m_Bbg; }

  // const Grid& local_grid() const { return m_local_grid; }
  // const Grid& local_grid_dual() const { return m_local_grid_dual; }
  // const Grid& super_grid() const { return m_super_grid; }
  // const Grid& data_grid() const { return m_data_grid; }
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
  void check_dev_mesh(Quadmesh& mesh);
  void check_dev_params(SimParams& params);

  // void save_snapshot(cu_sim_data& data);
  // void load_snapshot(cu_sim_data& data);
  void load_from_snapshot(const std::string& snapshot_file);

 private:
  // Environment() {}
  void setup_env();

  // cu_vector_field<Scalar> m_Ebg;
  // cu_vector_field<Scalar> m_Bbg;
};  // ----- end of class Environment -----
}  // namespace Aperture

#endif  // _SIM_ENVIRONMENT_DEV_H_
