#ifndef _SIM_ENVIRONMENT_DEV_H_
#define _SIM_ENVIRONMENT_DEV_H_

#include "commandline_args.h"
#include "config_file.h"
#include "core/grid.h"
#include "cuda/data/fields_dev.h"
#include "sim_environment.h"
#include "sim_params.h"
#include "utils/logger.h"
#include <memory>
#include <random>
#include <string>
#include <vector>
// #include "core/domain_info.h"
// #include "utils/hdf_exporter.h"
// #include "utils/mpi_comm.h"

namespace Aperture {

struct cu_sim_data;
// class DomainCommunicator;

////////////////////////////////////////////////////////////////////////////////
///  Class of the simulation environment. This class holds the basic
///  information that is useful for many other modules.
////////////////////////////////////////////////////////////////////////////////
class cu_sim_environment : public sim_environment {
 public:
  cu_sim_environment(int* argc, char*** argv);
  cu_sim_environment(const std::string& conf_file);
  ~cu_sim_environment();

  // Remove copy and assignment operators
  cu_sim_environment(cu_sim_environment const&) = delete;
  cu_sim_environment& operator=(cu_sim_environment const&) = delete;

  // void init_bg_fields(cu_sim_data& data);

  void check_dev_mesh(Quadmesh& mesh);
  void check_dev_params(SimParams& params);

  // void save_snapshot(cu_sim_data& data);
  // void load_snapshot(cu_sim_data& data);
  void load_from_snapshot(const std::string& snapshot_file);

  const std::vector<int>& dev_map() const { return m_dev_ids; }
  std::vector<int>& dev_map() { return m_dev_ids; }
  SimParams& sub_params(int i) { return m_sub_params[i]; }
  const SimParams& sub_params(int i) const { return m_sub_params[i]; }

 private:
  void setup_env();

  std::vector<int> m_dev_ids;
  std::vector<SimParams> m_sub_params;
};  // ----- end of class cu_sim_environment -----
}  // namespace Aperture

#endif  // _SIM_ENVIRONMENT_DEV_H_
