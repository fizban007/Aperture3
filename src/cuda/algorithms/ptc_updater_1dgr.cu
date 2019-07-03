#include "algorithms/ptc_updater_1dgr.h"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/data_ptrs.h"
#include "cuda/ptr_util.h"
#include "cuda/utils/interpolation.cuh"
#include "cuda/utils/pitchptr.cuh"
#include "sim_data.h"
#include "sim_environment.h"
#include <curand_kernel.h>

namespace Aperture {

namespace Kernels {

__global__ void
prepare_initial_condition(particle_data ptc,
                          Grid_1dGR::mesh_ptrs mesh_ptrs,
                          int multiplicity) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curandState local_state;
  curand_init(1234, id, 0, &local_state);
  for (int cell =
           blockIdx.x * blockDim.x + threadIdx.x + dev_mesh.guard[0];
       cell < dev_mesh.dims[0] - dev_mesh.guard[0];
       cell += blockDim.x * gridDim.x) {
    if (cell < dev_mesh.guard[0] ||
        cell > dev_mesh.dims[0] - dev_mesh.guard[0])
      continue;
    // if (cell != 350) continue;
    for (int n = 0; n < multiplicity; n++) {
      size_t idx = cell * multiplicity * 2 + n * 2;
      // ptc.x1[idx] = ptc.x1[idx + 1] = curand_uniform(&local_state);
      ptc.x1[idx] = ptc.x1[idx + 1] = 1.0f;
      ptc.p1[idx] = ptc.p1[idx + 1] = 0.0f;
      Scalar D1 = mesh_ptrs.D1[cell];
      Scalar D2 = mesh_ptrs.D2[cell];
      Scalar D3 = mesh_ptrs.D3[cell];
      Scalar alpha = mesh_ptrs.alpha[cell];
      Scalar K1 = mesh_ptrs.K1[cell];
      Scalar rho0 = mesh_ptrs.rho0[cell];
      Scalar u0 = sqrt(D2 / (D2 * (alpha * alpha - D3) + D1 * D1));

      float u = curand_uniform(&local_state);
      ptc.E[idx] = ptc.E[idx + 1] = u0;
      ptc.cell[idx] = ptc.cell[idx + 1] = cell;
      // ptc.cell[idx] = MAX_CELL;
      ptc.weight[idx] = 0.05 * dev_params.B0 * K1 +
                        std::abs(min(rho0 * K1, 0.0f)) / multiplicity;
      ptc.weight[idx + 1] = 0.05 * dev_params.B0 * K1 +
                            max(rho0 * K1, 0.0f) / multiplicity;
      // printf("p1 %f, x1 %f, u0 %f, w %f\n", ptc.p1[idx], ptc.x1[idx],
      // u0, ptc.weight[idx]);
      ptc.flag[idx] = set_ptc_type_flag(
          (u < dev_params.track_percent ? bit_or(ParticleFlag::tracked) : 0),
          ParticleType::electron);
      ptc.flag[idx + 1] = set_ptc_type_flag(
          (u < dev_params.track_percent ? bit_or(ParticleFlag::tracked) : 0),
          ParticleType::positron);
      ptc.id[idx] = atomicAdd(&dev_ptc_id, 1);
      ptc.id[idx + 1] = atomicAdd(&dev_ptc_id, 1);
    }
  }
}

__global__ void
prepare_initial_photons(photon_data photons,
                        Grid_1dGR::mesh_ptrs mesh_ptrs,
                        int multiplicity) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curandState local_state;
  curand_init(1234, id, 0, &local_state);
  for (int cell =
           blockIdx.x * blockDim.x + threadIdx.x + dev_mesh.guard[0];
       cell < dev_mesh.dims[0] - dev_mesh.guard[0];
       cell += blockDim.x * gridDim.x) {
    if (cell < dev_mesh.guard[0] ||
        cell > dev_mesh.dims[0] - dev_mesh.guard[0])
      continue;
    for (int n = 0; n < multiplicity; n++) {
      size_t idx = cell * multiplicity * 2 + n * 2;
      // photons.x1[idx] = photons.x1[idx + 1] =
      // curand_uniform(&local_state);
      photons.x1[idx] = photons.x1[idx + 1] = 1.0f;
      photons.p3[idx] = photons.p3[idx + 1] = 0.0f;
      // TODO: Compute photon E
      Scalar g11 = mesh_ptrs.gamma_rr[cell];
      Scalar g33 = mesh_ptrs.gamma_ff[cell];
      Scalar alpha = mesh_ptrs.alpha[cell];
      constexpr Scalar a = 0.99;
      Scalar xi = dev_mesh.pos(0, cell, 1.0f);
      const Scalar rp = 1.0f + std::sqrt(1.0f - a * a);
      const Scalar rm = 1.0f - std::sqrt(1.0f - a * a);
      Scalar exp_xi = std::exp(xi * (rp - rm));
      Scalar r = (rp - rm * exp_xi) / (1.0 - exp_xi);
      Scalar Delta = r * r - 2.0f * r + a * a;

      photons.p1[idx] = 10.0f * curand_uniform(&local_state) * Delta;
      photons.p1[idx + 1] =
          -10.0f * curand_uniform(&local_state) * Delta;
      photons.E[idx] = sgn(photons.p1[idx]) *
                       std::sqrt(g11 * square(photons.p1[idx] / Delta) +
                                 g33 * square(photons.p3[idx])) /
                       alpha;
      photons.E[idx + 1] =
          sgn(photons.p1[idx + 1]) *
          std::sqrt(g11 * square(photons.p1[idx + 1] / Delta) +
                    g33 * square(photons.p3[idx + 1])) /
          alpha;

      photons.weight[idx] = photons.weight[idx + 1] = 1.0f;
      photons.cell[idx] = photons.cell[idx + 1] = cell;
      photons.flag[idx] = bit_or(PhotonFlag::tracked);
      photons.flag[idx + 1] = bit_or(PhotonFlag::tracked);
    }
  }
}

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
update_ptc_1dgr(data_ptrs data, size_t num,
                Grid_1dGR::mesh_ptrs mesh_ptrs, Scalar dt, uint32_t step = 0) {
  auto& ptc = data.particles;
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = ptc.cell[idx];
    // printf("%u\n", c);
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
    //   printf("p1 %f, x1 %f, u0 %f, w %f\n", ptc.p1[idx], ptc.x1[idx],
    //   u0, ptc.weight[idx]);

    // interpolate quantities
    Scalar alpha =
        mesh_ptrs.alpha[c] * x1 + mesh_ptrs.alpha[c - 1] * nx1;
    Scalar D1 = mesh_ptrs.D1[c] * x1 + mesh_ptrs.D1[c - 1] * nx1;
    Scalar D2 = mesh_ptrs.D2[c] * x1 + mesh_ptrs.D2[c - 1] * nx1;
    Scalar D3 = mesh_ptrs.D3[c] * x1 + mesh_ptrs.D3[c - 1] * nx1;
    Scalar Dr = (x1 < 0.5f ? data.E1(c, 0) * (0.5f + x1) +
                                 data.E1(c - 1, 0) * (0.5f - x1)
                           : data.E1(c, 0) * (1.5f - x1) +
                                 data.E1(c + 1, 0) * (x1 - 0.5f));
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
    //   printf("gr_term %f, Er %f, q %f, m %f\n", gr_term, Er,
    //   dev_charges[sp], dev_masses[sp]);

    u0 = sqrt((D2 + p1 * p1) / (D2 * (alpha * alpha - D3) + D1 * D1));

    // printf("cell is %d, old p1 is %f, new p1 is %f, gr_term is %f\n",
    // c, ptc.p1[idx], p1, gr_term);
    ptc.p1[idx] = p1;

    vr = (p1 / u0 - D1) / D2;

    if ((step + 1) % dev_params.data_interval == 0 &&
        check_bit(flag, ParticleFlag::tracked)) {
      // Use p2 to store lower u_0
      Scalar theta = mesh_ptrs.theta[c] * x1 + mesh_ptrs.theta[c - 1] * nx1;
      Scalar Sigma = (r*r + a*a*square(std::cos(theta)));
      Scalar Delta = (r*r - 2.0f * r + a*a);
      Scalar g_00 = -(1.0f - 2.0f * r / Sigma);
      Scalar g_03 = -4.0 * r * a * square(std::sin(theta)) / Sigma;
      Scalar B31 = mesh_ptrs.B3B1[c] * x1 + mesh_ptrs.B3B1[c - 1] * nx1;

      Scalar u0inf = (g_00 + g_03 * dev_params.omega) * u0 + g_03 * B31 * Delta * u0 * vr;
      ptc.p2[idx] = sgn(vr) * std::abs(u0inf);
    }

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

        atomicAdd(&data.J1[offset + sizeof(Scalar)],
                  weight * djx / mesh_ptrs.K1_j[i + c + 1] *
                      dev_mesh.delta[0] / dt);
        // if (sp == 0)
        //   printf("j1 is %f, ",
        //          fields.J1[offset + sizeof(Scalar)]);

        atomicAdd(&data.Rho[sp][offset],
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
update_photon_1dgr(photon_data photons, size_t num,
                   Grid_1dGR::mesh_ptrs mesh_ptrs, Scalar dt, uint32_t step = 0) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    auto c = photons.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL || idx >= num) continue;
    if (!dev_mesh.is_in_bulk(c)) {
      photons.cell[idx] = MAX_CELL;
      continue;
    }

    auto p1 = photons.p1[idx];
    auto p3 = photons.p3[idx];
    auto x1 = photons.x1[idx];
    auto u0 = std::abs(photons.E[idx]);
    auto nx1 = 1.0f - x1;
    Scalar xi = dev_mesh.pos(0, c, x1);

    // Compute the physical r of the photon
    // FIXME: pass a in as a parameter
    constexpr Scalar a = 0.99;
    const Scalar rp = 1.0f + std::sqrt(1.0f - a * a);
    const Scalar rm = 1.0f - std::sqrt(1.0f - a * a);
    Scalar exp_xi = std::exp(xi * (rp - rm));
    Scalar r = (rp - rm * exp_xi) / (1.0 - exp_xi);

    // interpolate quantities
    Scalar alpha =
        mesh_ptrs.alpha[c] * x1 + mesh_ptrs.alpha[c - 1] * nx1;
    Scalar gamma11 =
        mesh_ptrs.gamma_rr[c] * x1 + mesh_ptrs.gamma_rr[c - 1] * nx1;
    Scalar gamma33 =
        mesh_ptrs.gamma_ff[c] * x1 + mesh_ptrs.gamma_ff[c - 1] * nx1;
    // Scalar D3 = mesh_ptrs.D3[c] * x1 + mesh_ptrs.D3[c - 1] * nx1;

    Scalar dDelta_dr = 2.0f * r - 2.0f;
    Scalar Delta_sqr = square(r * r - 2.0f * r + a * a);
    u0 = std::sqrt(gamma11 * p1 * p1 / Delta_sqr + gamma33 * p3 * p3) /
         alpha;
    // int c_0 = (x1 < 0.5f ? c - 1 : c);
    Scalar da2dr =
        (square(mesh_ptrs.alpha[c]) - square(mesh_ptrs.alpha[c - 1])) *
        dev_mesh.inv_delta[0];
    // Scalar da2dr = 2.0 * alpha * interp_deriv(x1, c, mesh_ptrs.alpha)
    // *
    //                dev_mesh.inv_delta[0];
    Scalar dgamma11dxi =
        (mesh_ptrs.gamma_rr[c] - mesh_ptrs.gamma_rr[c - 1]) *
        dev_mesh.inv_delta[0];
    Scalar dgamma33dxi =
        (mesh_ptrs.gamma_ff[c] - mesh_ptrs.gamma_ff[c - 1]) *
        dev_mesh.inv_delta[0];
    Scalar dbetadxi =
        (mesh_ptrs.beta_phi[c] - mesh_ptrs.beta_phi[c - 1]) *
        dev_mesh.inv_delta[0];

    // update momentum
    Scalar gr_term =
        (-0.5f * u0 * da2dr -
         (p1 * p1 * dgamma11dxi / Delta_sqr + p3 * p3 * dgamma33dxi) *
             0.5f / u0 +
         p3 * dbetadxi +
         gamma11 * p1 * p1 * dDelta_dr / (Delta_sqr * u0)) *
        dt;
    p1 += gr_term;

    u0 = std::sqrt(gamma11 * p1 * p1 / Delta_sqr + gamma33 * p3 * p3) /
         alpha;

    // printf("cell is %d, old p1 is %f, new p1 is %f, gr_term is %f\n",
    // c, ptc.p1[idx], p1, gr_term);
    photons.p1[idx] = p1;
    photons.E[idx] = sgn(p1) * u0;

    Scalar vr = gamma11 * p1 / (Delta_sqr * u0);

    if ((step + 1) % dev_params.data_interval == 0 &&
        check_bit(photons.flag[idx], PhotonFlag::tracked)) {
      // Use p2 to store lower u_0
      Scalar theta = mesh_ptrs.theta[c] * x1 + mesh_ptrs.theta[c - 1] * nx1;
      Scalar Sigma = (r*r + a*a*square(std::cos(theta)));
      Scalar Delta = (r*r - 2.0f * r + a*a);
      Scalar g_00 = -(1.0f + 2.0f * r * (r*r + a*a) / Sigma / Delta);
      Scalar g_03 = -2.0f * r * a / Sigma / Delta;

      Scalar u0inf = (u0 - g_03 * photons.p3[idx]) / g_00;
      photons.p2[idx] = sgn(vr) * std::abs(u0inf);
    }

    // compute movement
    Pos_t new_x1 = x1 + vr * dt * dev_mesh.inv_delta[0];
    // if (c > 160 && c < 190) {
    //   printf("new_x1 is %f\n", new_x1);
    // }
    int dc1 = floor(new_x1);
    if (dc1 > 1 || dc1 < -1) printf("Moving more than 1 cell!\n");
    new_x1 -= (Pos_t)dc1;

    photons.cell[idx] = c + dc1;
    photons.x1[idx] = new_x1;
  }
}

__global__ void filter_current1d(pitchptr<Scalar> j, pitchptr<Scalar> j_tmp,
                                 const Scalar *k1, bool boundary_lower,
                                 bool boundary_upper) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dev_mesh.dims[0];
       i += blockDim.x * gridDim.x) {
    size_t dx_plus =
        (boundary_upper && i == dev_mesh.dims[0] - dev_mesh.guard[0] - 1 ? 1
                                                                         : 0);
    size_t dx_minus = (boundary_lower && i == dev_mesh.guard[0] ? 1 : 0);
    j_tmp(i, 0) = 0.5f * j(i, 0) * k1[i];
    j_tmp(i, 0) += 0.25f * j(i + dx_plus, 0) * k1[i + dx_plus];
    j_tmp(i, 0) += 0.25f * j(i - dx_minus, 0) * k1[i - dx_minus];
    j_tmp(i, 0) /= k1[i];
  }
}

}  // namespace Kernels

ptc_updater_1dgr::ptc_updater_1dgr(sim_environment& env) : m_env(env) {
  m_tmp_j = multi_array<Scalar>(m_env.local_grid().extent());
}

ptc_updater_1dgr::~ptc_updater_1dgr() {}

void
ptc_updater_1dgr::update_particles(sim_data& data, double dt,
                                   uint32_t step) {
  Grid_1dGR* grid =
      dynamic_cast<Grid_1dGR*>(&m_env.local_grid());
  auto mesh_ptrs = grid->get_mesh_ptrs();
  auto data_p = get_data_ptrs(data);
  if (data.particles.number() > 0) {
    data.J.initialize();
    for (auto& rho : data.Rho) {
      rho.initialize();
    }

    Logger::print_info("Updating {} particles",
                       data.particles.number());
    Kernels::update_ptc_1dgr<<<2048, 512>>>(
        data_p, data.particles.number(), mesh_ptrs, dt, step);
    CudaCheckError();

    for (int i = 0; i < m_env.params().current_smoothing; i++) {
      Kernels::filter_current1d<<<256, 256>>>(
          get_pitchptr(data.J.data(0)), get_pitchptr(m_tmp_j), mesh_ptrs.K1_j, m_env.is_boundary(0),
          m_env.is_boundary(1));
      data.J.data(0).copy_from(m_tmp_j);

      for (int sp = 0; sp < m_env.params().num_species; sp++) {
        Kernels::filter_current1d<<<256, 256>>>(
            get_pitchptr(data.Rho[sp].data()), get_pitchptr(m_tmp_j), mesh_ptrs.K1,
            m_env.is_boundary(0), m_env.is_boundary(1));
        data.Rho[sp].data().copy_from(m_tmp_j);
      }
    }
  }

  if (data.photons.number() > 0) {
    Logger::print_info("Updating {} photons", data.photons.number());
    Kernels::update_photon_1dgr<<<2048, 512>>>(
        data.photons.data(), data.photons.number(), mesh_ptrs, dt, step);
    CudaCheckError();
  }
}

void
ptc_updater_1dgr::prepare_initial_condition(sim_data &data, int multiplicity) {
  const Grid_1dGR* g =
      dynamic_cast<const Grid_1dGR*>(&data.env.grid());
  if (g != nullptr) {
    Kernels::prepare_initial_condition<<<128, 256>>>(
        data.particles.data(), g->get_mesh_ptrs(), multiplicity);
    CudaCheckError();

    data.particles.set_num(g->mesh().reduced_dim(0) * 2 * multiplicity);
    // particles.append({0.5, 0.0, 4000})
  }
}

void
ptc_updater_1dgr::prepare_initial_photons(sim_data &data, int multiplicity) {
  const Grid_1dGR* g =
      dynamic_cast<const Grid_1dGR*>(&data.env.grid());
  if (g != nullptr) {
    Kernels::prepare_initial_photons<<<128, 256>>>(
        data.photons.data(), g->get_mesh_ptrs(), multiplicity);
    CudaCheckError();

    data.photons.set_num(g->mesh().reduced_dim(0) * 2 * multiplicity);
    // particles.append({0.5, 0.0, 4000})
  }
}

}  // namespace Aperture
