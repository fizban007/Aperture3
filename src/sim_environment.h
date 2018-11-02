#ifndef _SIM_ENVIRONMENT_H_
#define _SIM_ENVIRONMENT_H_

#include "commandline_args.h"
#include "config_file.h"
#include "sim_params.h"
#include <memory>
#include <random>
#include <string>
// #include "data/domain_info.h"
#include "data/grid.h"
#include "data/fields.h"
#include "utils/hdf_exporter.h"
// #include "metrics.h"
// #include "utils/data_exporter.h"
// #include "utils/mpi_comm.h"
#include "utils/logger.h"
// #include "cuda/constant_mem.h"
// #include "boundary_conditions.h"
// #include "initial_conditions.h"

namespace Aperture {

// struct SimData;
// class DomainCommunicator;

////////////////////////////////////////////////////////////////////////////////
///  Class of the simulation environment. This class holds the basic
///  information that is useful for many other modules.
////////////////////////////////////////////////////////////////////////////////
class Environment {
 public:
  Environment(int* argc, char*** argv);
  Environment(const std::string& conf_file);
  ~Environment();

  // Remove copy and assignment operators
  Environment(Environment const&) = delete;
  Environment& operator=(Environment const&) = delete;

  // void set_initial_condition(SimData& data);
  // void set_initial_condition(SimData& data, const Index& start, const
  // Extent& extent); void setup_domain(int num_nodes); void
  // setup_domain(int dimx, int dimy, int dimz = 1); void
  // setup_local_grid(Grid& local_grid, const Grid& super_grid,
  //                       const DomainInfo& info);

  // void set_initial_condition(InitialCondition* ic);
  // void add_fieldBC(fieldBC* bc);
  // void add_ptcBC(ptcBC* bc);

  void add_output(const std::string& name);

  // void apply_initial_condition(SimData& data);

  float gen_rand() { return m_dist(m_generator); }

  // data access methods
  const CommandArgs& args() const { return m_args; }
  // const SimParams& conf() const { return m_conf_file.data(); }
  SimParams& params() { return m_params; }
  const SimParams& params() const { return m_params; }
  // const ConfigFile& conf_file() const { return m_conf_file; }
  const Grid& grid() const { return *m_grid; }
  const Grid& local_grid() const { return *m_grid; }
  const Quadmesh& mesh() const { return m_grid->mesh(); }
  VectorField<Scalar>& E_bg() { return m_Ebg; }
  VectorField<Scalar>& B_bg() { return m_Bbg; }

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

 private:
  // Environment() {}
  void setup_env(const std::string& conf_file);

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
  VectorField<Scalar> m_Ebg;
  VectorField<Scalar> m_Bbg;

  // std::unique_ptr<MPIComm> m_comm;
  // std::unique_ptr<DataExporter> m_exporter;
  // std::unique_ptr<InitialCondition> m_ic;
  std::default_random_engine m_generator;
  std::uniform_real_distribution<float> m_dist;

};  // ----- end of class sim_environment -----
}  // namespace Aperture

#endif  // _SIM_ENVIRONMENT_H_
