#ifndef _CU_SIM_ENVIRONMENT_H_
#define _CU_SIM_ENVIRONMENT_H_

#include "commandline_args.h"
#include "config_file.h"
#include "core/grid.h"
#include "cuda/data/cu_multi_array.h"
#include "cuda/data/fields_dev.h"
#include "sim_environment.h"
#include "sim_params.h"
#include "utils/logger.h"
#include <cuda_runtime.h>
#include <memory>
#include <random>
#include <string>
#include <vector>
// #include "core/domain_info.h"
// #include "utils/mpi_comm.h"

namespace Aperture {

struct cu_sim_data;

///  Class of the simulation environment. This class holds the basic
///  information that is useful for many other modules.
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

  // void get_sub_guard_cells(
  //     std::vector<cu_vector_field<Scalar>>& field) const;
  // void get_sub_guard_cells(
  //     std::vector<cu_scalar_field<Scalar>>& field) const;
  // void send_sub_guard_cells(
  //     std::vector<cu_vector_field<Scalar>>& field) const;
  // void send_sub_guard_cells(
  //     std::vector<cu_scalar_field<Scalar>>& field) const;

  int dev_id() const { return m_dev_id; }

 private:
  void setup_env();

  // void get_sub_guard_cells_left(cudaPitchedPtr p_src,
  //                               cudaPitchedPtr p_dst,
  //                               const Quadmesh& mesh_src,
  //                               const Quadmesh& mesh_dst, int src_dev,
  //                               int dst_dev) const;
  // void get_sub_guard_cells_right(cudaPitchedPtr p_src,
  //                                cudaPitchedPtr p_dst,
  //                                const Quadmesh& mesh_src,
  //                                const Quadmesh& mesh_dst, int src_dev,
  //                                int dst_dev) const;

  // void send_sub_guard_cells_left(cu_multi_array<Scalar>& src,
  //                                cu_multi_array<Scalar>& dst,
  //                                const Quadmesh& mesh_src,
  //                                const Quadmesh& mesh_dst,
  //                                int buffer_id, int src_dev,
  //                                int dst_dev, bool stagger) const;

  // void send_sub_guard_cells_right(cu_multi_array<Scalar>& src,
  //                                 cu_multi_array<Scalar>& dst,
  //                                 const Quadmesh& mesh_src,
  //                                 const Quadmesh& mesh_dst,
  //                                 int buffer_id, int src_dev,
  //                                 int dst_dev, bool stagger) const;
  // void add_from_buffer_left(cu_multi_array<Scalar>& dst,
  //                           const Quadmesh& mesh_dst, int buffer_id,
  //                           bool stagger) const;
  // void add_from_buffer_right(cu_multi_array<Scalar>& dst,
  //                            const Quadmesh& mesh_dst, int buffer_id,
  //                            bool stagger) const;

  // std::vector<cu_multi_array<Scalar>> m_sub_buffer_left;
  // std::vector<cu_multi_array<Scalar>> m_sub_buffer_right;
  int m_dev_id;
};  // ----- end of class cu_sim_environment -----
}  // namespace Aperture

#endif  // _CU_SIM_ENVIRONMENT_H_
