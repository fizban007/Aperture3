#include "core/detail/multi_array_utils.hpp"
#include "cuda/constant_mem.h"
#include "cuda/core/cu_sim_data.h"
#include "cuda/core/ptc_updater_dev.h"
#include "cuda/core/ptc_updater_helper.cuh"
#include "cuda/core/cu_sim_environment.h"
#include "cuda/cudaUtility.h"
#include "cuda/kernels.h"
#include "cuda/ptr_util.h"
#include "cuda/utils/interpolation.cuh"
#include "cuda/utils/iterate_devices.h"
#include "utils/logger.h"
#include "utils/util_functions.h"

#define DEPOSIT_EPS 1.0e-10f

namespace Aperture {

namespace Kernels {

__global__ void
check_dev_fields(cudaPitchedPtr p) {
  printf("%lu, %lu, %lu\n", p.pitch, p.xsize, p.ysize);
}

}  // namespace Kernels

PtcUpdaterDev::PtcUpdaterDev(const cu_sim_environment &env)
    : m_env(env) {
  // m_extent = m_env.grid().extent();
  unsigned int num_devs = env.dev_map().size();
  m_dev_fields.resize(num_devs);
  // m_extent.resize(num_devs);
  // FIXME: select the correct device?
  for (unsigned int n = 0; n < num_devs; n++) {
    int dev_id = env.dev_map()[n];
    CudaSafeCall(cudaSetDevice(dev_id));
    CudaSafeCall(cudaMallocManaged(
        &m_dev_fields[n].Rho,
        // m_env.params().num_species * sizeof(cudaPitchedPtr)));
        m_env.params().num_species * sizeof(typed_pitchedptr<Scalar>)));
  }
  m_fields_initialized = false;
}

PtcUpdaterDev::~PtcUpdaterDev() {
  for (unsigned int n = 0; n < m_dev_fields.size(); n++) {
    int dev_id = m_env.dev_map()[n];
    CudaSafeCall(cudaSetDevice(dev_id));
    CudaSafeCall(cudaFree(m_dev_fields[n].Rho));
  }
}

void
PtcUpdaterDev::initialize_dev_fields(cu_sim_data &data) {
  if (!m_fields_initialized) {
    for_each_device(data.dev_map, [this, &data](int n) {
      m_dev_fields[n].E1 = data.E[n].ptr(0);
      m_dev_fields[n].E2 = data.E[n].ptr(1);
      m_dev_fields[n].E3 = data.E[n].ptr(2);
      m_dev_fields[n].B1 = data.B[n].ptr(0);
      m_dev_fields[n].B2 = data.B[n].ptr(1);
      m_dev_fields[n].B3 = data.B[n].ptr(2);
      m_dev_fields[n].J1 = data.J[n].ptr(0);
      m_dev_fields[n].J2 = data.J[n].ptr(1);
      m_dev_fields[n].J3 = data.J[n].ptr(2);
      for (int i = 0; i < data.num_species; i++) {
        m_dev_fields[n].Rho[i] = data.Rho[i][n].ptr();
      }
    });
  }
  m_fields_initialized = true;
}

// void
// PtcUpdaterDev::update_particles(cu_sim_data &data, double dt,
//                                 uint32_t step) {
//   Logger::print_info("Updating particles");
//   // Track the right fields
//   initialize_dev_fields(data);

//   if (m_env.grid().dim() == 1) {
//     Kernels::update_particles_1d<<<512, 512>>>(
//         data.particles.data(), data.particles.number(),
//         (const Scalar *)data.E.ptr(0).ptr, (Scalar
//         *)data.J.ptr(0).ptr, (Scalar *)data.Rho[0].ptr().ptr, dt);
//     CudaCheckError();
//   } else if (m_env.grid().dim() == 2) {
//     Kernels::update_particles_2d<<<256, 256>>>(data.particles.data(),
//                                                data.particles.number(),
//                                                m_dev_fields, dt);
//     CudaCheckError();
//   } else if (m_env.grid().dim() == 3) {
//     Kernels::update_particles<<<256, 128>>>(data.particles.data(),
//                                             data.particles.number(),
//                                             m_dev_fields, dt);
//     CudaCheckError();

//     Kernels::deposit_current_3d<<<256, 256>>>(data.particles.data(),
//                                               data.particles.number(),
//                                               m_dev_fields, dt);
//     CudaCheckError();
//   }
//   cudaDeviceSynchronize();
// }

// void
// PtcUpdaterDev::handle_boundary(cu_sim_data &data) {
//   // erase_ptc_in_guard_cells(data.particles.data().cell,
//   //                          data.particles.number());
// }

}  // namespace Aperture
