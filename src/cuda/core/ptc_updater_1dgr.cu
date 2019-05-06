#include "cuda/constant_mem.h"
#include "cuda/core/cu_sim_data1d.h"
#include "cuda/core/cu_sim_environment.h"
#include "cuda/core/ptc_updater_1dgr.h"
#include "cuda/cudaUtility.h"
#include "cuda/ptr_util.h"
#include "cuda/utils/interpolation.cuh"

namespace Aperture {

namespace Kernels {

__device__ Scalar
interp_deriv(Scalar x, int c, const Scalar* array) {
  Scalar d0 = (x < 0.5f ? array[c - 1] - array[c - 2]
                        : array[c] - array[c - 1]);
  Scalar d1 =
      (x < 0.5f ? array[c] - array[c - 1] : array[c + 1] - array[c]);
  return (x < 0.5f ? (0.5f + x) * d1 + (0.5f - x) * d0
                   : (x - 0.5f) * d1 + (1.5f - x) * d0);
}

__global__ void
update_ptc_1dgr(particle1d_data ptc, size_t num,
                ptc_updater_1dgr_dev::fields_data fields,
                Grid_1dGR_dev::mesh_ptrs mesh_ptrs, Scalar dt) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = ptc.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL || idx >= num) continue;
    if (!dev_mesh.is_in_bulk(c)) {
      ptc.cell[idx] = MAX_CELL;
      continue;
    }

    auto p1 = ptc.p1[idx];
    auto x1 = ptc.x1[idx];
    auto u0 = ptc.E[idx];
    auto nx1 = 1.0f - x1;
    auto flag = ptc.flag[idx];
    int sp = get_ptc_type(flag);
    Scalar xi = dev_mesh.pos(0, c, x1);
    // FIXME: pass a in as a parameter
    constexpr Scalar a = 0.99;
    const Scalar rp = 1.0f + std::sqrt(1.0f - a * a);
    const Scalar rm = 1.0f - std::sqrt(1.0f - a * a);
    Scalar exp_xi = std::exp(xi * (rp - rm));
    Scalar r = (rp - rm * exp_xi) / (1.0 - exp_xi);
    // if (idx == 10)
    //   printf("p1 %f, x1 %f, u0 %f, w %f\n", ptc.p1[idx], ptc.x1[idx], u0, ptc.weight[idx]);

    // interpolate quantities
    Scalar alpha =
        mesh_ptrs.alpha[c] * x1 + mesh_ptrs.alpha[c - 1] * nx1;
    Scalar D1 = mesh_ptrs.D1[c] * x1 + mesh_ptrs.D1[c - 1] * nx1;
    Scalar D2 = mesh_ptrs.D2[c] * x1 + mesh_ptrs.D2[c - 1] * nx1;
    Scalar D3 = mesh_ptrs.D3[c] * x1 + mesh_ptrs.D3[c - 1] * nx1;
    Scalar Dr = (x1 < 0.5f ? fields.E1(c, 0) * (0.5f + x1) +
                                 fields.E1(c - 1, 0) * (0.5f - x1)
                           : fields.E1(c, 0) * (1.5f - x1) +
                                 fields.E1(c + 1, 0) * (x1 - 0.5f));
    // Scalar Dphi = fields.E3[c] * x1 + fields.E3[c - 1] * nx1;
    // Scalar Dphi = *ptrAddr(fields.E3, c, 0) * x1 +
    //               *ptrAddr(fields.E3, c - 1, 0) * nx1;
    Scalar agrr = mesh_ptrs.agrr[c] * x1 + mesh_ptrs.agrr[c - 1] * nx1;
    // Scalar agrf = mesh_ptrs.agrf[c] * x1 + mesh_ptrs.agrf[c - 1] *
    // nx1; Scalar Er = agrr * Dr + agrf * Dphi;
    Scalar Er = agrr * Dr;

    Scalar vr = (p1 / u0 - D1) / D2;
    // int c_0 = (x1 < 0.5f ? c - 1 : c);
    Scalar da2dr =
        (square(mesh_ptrs.alpha[c]) - square(mesh_ptrs.alpha[c - 1])) *
        dev_mesh.inv_delta[0];
    // Scalar da2dr = 2.0 * alpha * interp_deriv(x1, c, mesh_ptrs.alpha)
    // *
    //                dev_mesh.inv_delta[0];
    Scalar dD1dr =
        (mesh_ptrs.D1[c] - mesh_ptrs.D1[c - 1]) * dev_mesh.inv_delta[0];
    Scalar dD2dr =
        (mesh_ptrs.D2[c] - mesh_ptrs.D2[c - 1]) * dev_mesh.inv_delta[0];
    Scalar dD3dr =
        (mesh_ptrs.D3[c] - mesh_ptrs.D3[c - 1]) * dev_mesh.inv_delta[0];
    Scalar dDelta_dr = 2.0f * r - 2.0;

    // update momentum
    Scalar gr_term =
        -0.5f * u0 *
            (da2dr - (dD3dr + 2.0f * (dD1dr - D1 * dDelta_dr) * vr +
                      (dD2dr - 2.0f * D2 * dDelta_dr) * vr * vr)) *
            dt +
        dDelta_dr * p1 * vr * dt;
    p1 += gr_term;
    Scalar E_term = Er * dev_charges[sp] / dev_masses[sp] * dt;
    p1 += E_term;
    // if (idx == 10)
    //   printf("gr_term %f, Er %f, q %f, m %f\n", gr_term, Er, dev_charges[sp], dev_masses[sp]);


    u0 = sqrt((D2 + p1 * p1) / (D2 * (alpha * alpha - D3) + D1 * D1));

    // printf("cell is %d, old p1 is %f, new p1 is %f, gr_term is %f\n",
    // c, ptc.p1[idx], p1, gr_term);
    ptc.p1[idx] = p1;

    vr = (p1 / u0 - D1) / D2;

    // compute movement
    Pos_t new_x1 = x1 + vr * dt * dev_mesh.inv_delta[0];
    // if (c > 160 && c < 190) {
    //   printf("new_x1 is %f\n", new_x1);
    // }
    int dc1 = floor(new_x1);
    if (dc1 > 1 || dc1 < -1) printf("Moving more than 1 cell!\n");
    new_x1 -= (Pos_t)dc1;

    ptc.cell[idx] = c + dc1;
    ptc.x1[idx] = new_x1;

    nx1 = 1.0f - new_x1;
    alpha = mesh_ptrs.alpha[c + dc1] * new_x1 +
            mesh_ptrs.alpha[c + dc1 - 1] * nx1;
    D1 = mesh_ptrs.D1[c + dc1] * new_x1 +
         mesh_ptrs.D1[c + dc1 - 1] * nx1;
    D2 = mesh_ptrs.D2[c + dc1] * new_x1 +
         mesh_ptrs.D2[c + dc1 - 1] * nx1;
    D3 = mesh_ptrs.D3[c + dc1] * new_x1 +
         mesh_ptrs.D3[c + dc1 - 1] * nx1;

    // compute energy at the new position
    u0 = sqrt((D2 + p1 * p1) / (D2 * (alpha * alpha - D3) + D1 * D1));
    ptc.E[idx] = u0;
    // printf("u0 is %f, p1 is %f, vr is %f\n", u0, p1, vr);

    // TODO: deposit current
    if (!check_bit(flag, ParticleFlag::ignore_current)) {
      Spline::cloud_in_cell interp;
      Scalar weight = -dev_charges[sp] * ptc.weight[idx];
      int i_0 = (dc1 == -1 ? -2 : -1);
      int i_1 = (dc1 == 1 ? 1 : 0);
      Scalar djx = 0.0f;
      for (int i = i_0; i <= i_1; i++) {
        Scalar sx0 = interp(-x1 + i + 1);
        Scalar sx1 = interp(-new_x1 + (i + 1 - dc1));
        int offset = (i + c) * sizeof(Scalar);
        djx += sx1 - sx0;
        // printf("djx%d is %f, ", i, sx1 - sx0);

        atomicAdd(&fields.J1[offset + sizeof(Scalar)],
                  weight * djx / mesh_ptrs.K1_j[i + c + 1] *
                      dev_mesh.delta[0] / dt);
        // if (sp == 0)
        //   printf("j1 is %f, ",
        //          fields.J1[offset + sizeof(Scalar)]);

        atomicAdd(&fields.Rho[sp][offset],
                  -weight * sx1 / mesh_ptrs.K1[i + c]);
        // if (sp == 0)
        //   printf("sx1 is %f\n", sx1);
      }
      // printf("j1 is %f at %d\n", *ptrAddr(fields.J1, c, 0), c);
      // printf("\n");
      // if (c > 330 && c < 370) {
      //   printf("j1 at %d is %f, vr is %f, p1 is %f, u0 is %f\n",
      //          c + dc1, *ptrAddr(fields.J1, c, 0), vr, p1, u0);
      // }
    }
  }
}

__global__ void
update_photon_1dgr(photon1d_data photons, size_t num,
                   Grid_1dGR_dev::mesh_ptrs mesh_ptrs, Scalar dt) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = photons.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL || idx >= num) continue;
  }
}

}  // namespace Kernels

ptc_updater_1dgr_dev::ptc_updater_1dgr_dev(
    const cu_sim_environment& env)
    : m_env(env) {
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
                                       uint32_t step) {
  initialize_dev_fields(data);
  if (data.particles.number() > 0) {
    data.J.initialize();
    for (auto& rho : data.Rho) {
      rho.initialize();
    }

    const Grid_1dGR_dev& grid =
        *dynamic_cast<const Grid_1dGR_dev*>(data.grid.get());
    auto mesh_ptrs = grid.get_mesh_ptrs();
    Logger::print_info("Updating {} particles",
                       data.particles.number());
    Kernels::update_ptc_1dgr<<<256, 512>>>(data.particles.data(),
                                           data.particles.number(),
                                           m_dev_fields, mesh_ptrs, dt);
    CudaCheckError();
  }
}

void
ptc_updater_1dgr_dev::initialize_dev_fields(cu_sim_data1d& data) {
  if (!m_fields_initialized) {
    m_dev_fields.E1 = data.E.ptr(0);
    m_dev_fields.E3 = data.E.ptr(2);
    // m_dev_fields.B1 = data.B.ptr(0);
    // m_dev_fields.B3 = data.B.ptr(2);
    m_dev_fields.J1 = data.J.ptr(0);
    m_dev_fields.J3 = data.J.ptr(2);
    for (int i = 0; i < m_env.params().num_species; i++) {
      m_dev_fields.Rho[i] = data.Rho[i].ptr();
    }
    m_fields_initialized = true;
  }
}

}  // namespace Aperture
