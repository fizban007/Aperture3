#include "core/detail/multi_array_utils.hpp"
#include "cuda/constant_mem.h"
#include "cuda/core/cu_sim_data.h"
#include "cuda/core/cu_sim_environment.h"
#include "cuda/core/ptc_updater_dev.h"
#include "cuda/core/ptc_updater_helper.cuh"
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
  // m_extent.resize(num_devs);
  CudaSafeCall(cudaMallocManaged(
      &m_dev_fields.Rho,
      // m_env.params().num_species * sizeof(cudaPitchedPtr)));
      m_env.params().num_species * sizeof(pitchptr<Scalar>)));
  m_fields_initialized = false;
}

PtcUpdaterDev::~PtcUpdaterDev() {
  CudaSafeCall(cudaFree(m_dev_fields.Rho));
}

void
PtcUpdaterDev::initialize_dev_fields(cu_sim_data &data) {
  if (!m_fields_initialized) {
    m_dev_fields.E1 = data.E.ptr(0);
    m_dev_fields.E2 = data.E.ptr(1);
    m_dev_fields.E3 = data.E.ptr(2);
    m_dev_fields.B1 = data.B.ptr(0);
    m_dev_fields.B2 = data.B.ptr(1);
    m_dev_fields.B3 = data.B.ptr(2);
    m_dev_fields.J1 = data.J.ptr(0);
    m_dev_fields.J2 = data.J.ptr(1);
    m_dev_fields.J3 = data.J.ptr(2);
    for (int i = 0; i < data.num_species; i++) {
      m_dev_fields.Rho[i] = data.Rho[i].ptr();
    }
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
