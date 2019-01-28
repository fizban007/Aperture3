#include "cuda/constant_mem.h"
#include "cuda/core/cu_sim_data1d.h"
#include "cuda/core/ptc_updater_1dgr.h"
#include "cuda/core/sim_environment_dev.h"
#include "cuda/cudaUtility.h"

namespace Aperture {

namespace Kernels {

__global__ void
update_ptc_1dgr(particle1d_data ptc, size_t num,
                ptc_updater_1dgr_dev::fields_data fields,
                Grid_1dGR_dev::mesh_ptrs mesh_ptrs, Scalar dt) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = ptc.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL || idx >= num) continue;

    auto p1 = ptc.p1[idx];
    auto x1 = ptc.x1[idx];
    auto u0 = ptc.E[idx];
    auto nx1 = 1.0f - x1;
    auto flag = ptc.flag[idx];
    int sp = get_ptc_type(flag);

    // interpolate quantities
    Scalar a2 =
        mesh_ptrs.alpha2[c] * x1 + mesh_ptrs.alpha2[c - 1] * nx1;
    Scalar D1 = mesh_ptrs.D1[c] * x1 + mesh_ptrs.D1[c - 1] * nx1;
    Scalar D2 = mesh_ptrs.D2[c] * x1 + mesh_ptrs.D2[c - 1] * nx1;
    Scalar D3 = mesh_ptrs.D3[c] * x1 + mesh_ptrs.D3[c - 1] * nx1;
    Scalar Dr = fields.E1[c] * x1 + fields.E1[c - 1] * nx1;
    Scalar Dphi = fields.E3[c] * x1 + fields.E3[c - 1] * nx1;
    Scalar agrr = mesh_ptrs.agrr[c] * x1 + mesh_ptrs.agrr[c - 1] * nx1;
    Scalar agrf = mesh_ptrs.agrf[c] * x1 + mesh_ptrs.agrf[c - 1] * nx1;
    Scalar Er = agrr * Dr + agrf * Dphi;

    Scalar vr = (p1 / u0 - D1) / D2;
    Scalar da2dr = (mesh_ptrs.alpha2[c] - mesh_ptrs.alpha2[c - 1]) *
                   dev_mesh.inv_delta[0];
    Scalar dD1dr =
        (mesh_ptrs.D1[c] - mesh_ptrs.D1[c - 1]) * dev_mesh.inv_delta[0];
    Scalar dD2dr =
        (mesh_ptrs.D2[c] - mesh_ptrs.D2[c - 1]) * dev_mesh.inv_delta[0];
    Scalar dD3dr =
        (mesh_ptrs.D3[c] - mesh_ptrs.D3[c - 1]) * dev_mesh.inv_delta[0];

    // update momentum
    p1 -= 0.5f * u0 *
          (da2dr - (dD3dr + 2.0f * dD1dr * vr + dD2dr * vr * vr)) * dt;
    p1 += Er * dt * dev_charges[sp] / dev_masses[sp];

    u0 = sqrt((1.0f + p1 * p1 / D2) / (a2 - D3 + D1 * D1 / D2));

    ptc.p1[idx] = p1;

    vr = (p1 / u0 - D1) / D2;

    // compute movement
    Pos_t new_x1 = x1 + vr * dt * dev_mesh.inv_delta[0];
    int dc1 = floor(new_x1);
    ptc.cell[idx] = c + dc1;
    new_x1 -= (Pos_t)dc1;

    ptc.x1[idx] = new_x1;

    nx1 = 1.0f - new_x1;
    a2 = mesh_ptrs.alpha2[c + dc1] * new_x1 +
         mesh_ptrs.alpha2[c + dc1 - 1] * nx1;
    D1 = mesh_ptrs.D1[c + dc1] * new_x1 +
         mesh_ptrs.D1[c + dc1 - 1] * nx1;
    D2 = mesh_ptrs.D2[c + dc1] * new_x1 +
         mesh_ptrs.D2[c + dc1 - 1] * nx1;
    D3 = mesh_ptrs.D3[c + dc1] * new_x1 +
         mesh_ptrs.D3[c + dc1 - 1] * nx1;

    // compute energy at the new position
    u0 = sqrt((1.0f + p1 * p1 / D2) / (a2 - D3 + D1 * D1 / D2));
    ptc.E[idx] = u0;

    // TODO: deposit current
  }
}

}  // namespace Kernels

ptc_updater_1dgr_dev::ptc_updater_1dgr_dev(
    const cu_sim_environment& env)
    : m_env(env) {
  const Grid_1dGR_dev& grid =
      dynamic_cast<const Grid_1dGR_dev&>(env.grid());
  // TODO: Check error!!
  m_mesh_ptrs = grid.get_mesh_ptrs();

  CudaSafeCall(cudaMallocManaged(
      &m_dev_fields.Rho,
      m_env.params().num_species * sizeof(cudaPitchedPtr)));
  m_fields_initialized = false;
}

ptc_updater_1dgr_dev::~ptc_updater_1dgr_dev() {
  CudaSafeCall(cudaFree(m_dev_fields.Rho));
}

void
ptc_updater_1dgr_dev::update_particles(cu_sim_data1d& data, double dt,
                                       uint32_t step) {}

void
ptc_updater_1dgr_dev::initialize_dev_fields(cu_sim_data1d& data) {
  if (!m_fields_initialized) {
    m_dev_fields.E1 = data.E.ptr(0);
    m_dev_fields.E3 = data.E.ptr(2);
    m_dev_fields.B1 = data.B.ptr(0);
    m_dev_fields.B3 = data.B.ptr(2);
    m_dev_fields.J1 = data.J.ptr(0);
    m_dev_fields.J3 = data.J.ptr(2);
    for (int i = 0; i < m_env.params().num_species; i++) {
      m_dev_fields.Rho[i] = data.Rho[i].ptr();
    }
    m_fields_initialized = true;
  }
}

}  // namespace Aperture