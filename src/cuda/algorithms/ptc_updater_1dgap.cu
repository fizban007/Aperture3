#include "algorithms/ptc_updater_1dgap.h"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/data_ptrs.h"
#include "cuda/ptr_util.h"
#include "cuda/utils/interpolation.cuh"
#include "cuda/utils/pitchptr.h"
#include "sim_data.h"
#include "sim_environment.h"
#include <curand_kernel.h>

namespace Aperture {

namespace Kernels {

HD_INLINE Scalar
beta_phi(Scalar x) {
  return x;
}

__global__ void
prepare_initial_condition(particle_data ptc, Scalar rho0,
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
      float ux = curand_uniform(&local_state);
      ptc.x1[idx] = ptc.x1[idx + 1] = ux;
      Scalar x_pos = dev_mesh.pos(0, cell, ux);
      // ptc.x1[idx] = ptc.x1[idx + 1] = 1.0f;
      ptc.p1[idx] = ptc.p1[idx + 1] = 0.0f;

      Scalar beta = beta_phi(dev_mesh.pos(0, cell, ptc.x1[idx]));
      ptc.E[idx] = ptc.E[idx + 1] = sqrt(1.0f + beta * beta);
      ptc.cell[idx] = ptc.cell[idx + 1] = cell;

      Scalar rho = rho0 * atan(2.0f * x_pos);
      ptc.weight[idx] = 2.0f * dev_params.B0 / multiplicity +
                        std::abs(min(rho, 0.0f)) / multiplicity;
      ptc.weight[idx + 1] = 2.0f * dev_params.B0 / multiplicity +
                            max(rho, 0.0f) / multiplicity;

      float u = curand_uniform(&local_state);
      if (u < dev_params.track_percent) {
        ptc.flag[idx] = set_ptc_type_flag(bit_or(ParticleFlag::tracked),
                                          ParticleType::electron);
        ptc.id[idx] = atomicAdd(&dev_ptc_id, 1);
      } else {
        ptc.flag[idx] = set_ptc_type_flag(0, ParticleType::electron);
      }
      u = curand_uniform(&local_state);
      if (u < dev_params.track_percent) {
        ptc.flag[idx + 1] = set_ptc_type_flag(
            bit_or(ParticleFlag::tracked), ParticleType::positron);
        ptc.id[idx + 1] = atomicAdd(&dev_ptc_id, 1);
      } else {
        ptc.flag[idx + 1] =
            set_ptc_type_flag(0, ParticleType::positron);
      }
    }
  }
}

__device__ Scalar interp_deriv(Scalar x, int c, const Scalar* array);

__global__ void
update_ptc_1dgap(data_ptrs data, size_t num, Scalar dt,
                 uint32_t step = 0) {
  auto& ptc = data.particles;
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num;
       idx += blockDim.x * gridDim.x) {
    if (idx >= num) continue;
    auto c = ptc.cell[idx];
    // Skip empty particles
    if (c == MAX_CELL) continue;
    if (!dev_mesh.is_in_bulk(c)) {
      ptc.cell[idx] = MAX_CELL;
      continue;
    }

    auto p1 = ptc.p1[idx];
    auto x1 = ptc.x1[idx];
    auto gamma = ptc.E[idx];
    auto flag = ptc.flag[idx];
    int sp = get_ptc_type(flag);
    Scalar xi = dev_mesh.pos(0, c, x1);
    Scalar beta = beta_phi(dev_mesh.pos(0, c, x1));
    // printf("cell is %d, old p1 is %f, new x1 is %f, u0 is %f\n",
    //        c, p1, x1, u0);

    Scalar Dr = (x1 < 0.5f ? data.E3(c, 0) * (0.5f + x1) +
                                 data.E3(c - 1, 0) * (0.5f - x1)
                           : data.E3(c, 0) * (1.5f - x1) +
                                 data.E3(c + 1, 0) * (x1 - 0.5f));

    Scalar E_term = Dr * dev_charges[sp] / dev_masses[sp] * dt;
    p1 += E_term;
    double f =
        (gamma - (beta < 0.0 ? -1.0 : 1.0) * p1) / (1.0 + beta * beta);
    p1 += (beta / gamma) * f * f * dt;

    ptc.p1[idx] = p1;
    gamma = std::sqrt(1.0f + p1 * p1 + beta * beta);
    Scalar vr = ((beta < 0.0 ? -1.0 : 1.0) * p1 / gamma + beta * beta) /
                (1.0 + beta * beta);
    // printf("vr is %f\n", vr);
    if (beta < 0.0f) vr *= -1.0f;

    // compute movement
    Pos_t new_x1 = x1 + vr * dt * dev_mesh.inv_delta[0];
    int dc1 = floor(new_x1);
    if (dc1 > 1 || dc1 < -1) {
      printf("Moving more than 1 cell! vr is %f\n", vr);
    }
    new_x1 -= (Pos_t)dc1;

    ptc.cell[idx] = c + dc1;
    ptc.x1[idx] = new_x1;

    beta = beta_phi(dev_mesh.pos(0, c + dc1, new_x1));
    // compute energy at the new position
    gamma = sqrt(1.0f + p1 * p1 + beta * beta);
    ptc.E[idx] = gamma;
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
                  weight * djx * dev_mesh.delta[0] / dt);
        // if (sp == 0)
        //   printf("j1 is %f, ",
        //          data.J1[offset + sizeof(Scalar)]);

        atomicAdd(&data.Rho[sp][offset], -weight * sx1);
        // if (sp == 0)
        //   printf("sx1 is %f\n", sx1);
      }
    }
  }
}

__global__ void
update_photon_1dgap(photon_data photons, size_t num, Scalar dt,
                    uint32_t step = 0) {
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
    auto x1 = photons.x1[idx];

    // compute movement
    Pos_t new_x1 = x1 + sgn(p1) * dt * dev_mesh.inv_delta[0];
    // if (c > 160 && c < 190) {
    //   printf("new_x1 is %f\n", new_x1);
    // }
    int dc1 = floor(new_x1);
    if (dc1 > 1 || dc1 < -1) printf("Moving more than 1 cell!\n");
    new_x1 -= (Pos_t)dc1;

    photons.cell[idx] = c + dc1;
    photons.x1[idx] = new_x1;
    // photons.path_left[idx] -= dt;
  }
}

__global__ void
filter_current1d(pitchptr<Scalar> j, pitchptr<Scalar> j_tmp,
                 bool boundary_lower, bool boundary_upper) {
  // for (size_t i = blockIdx.x * blockDim.x + threadIdx.x +
  // dev_mesh.guard[0];
  //      i < dev_mesh.dims[0] - dev_mesh.guard[0]; i += blockDim.x *
  //      gridDim.x) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < dev_mesh.dims[0]; i += blockDim.x * gridDim.x) {
    size_t dx_plus =
        (boundary_upper && i == dev_mesh.dims[0] - dev_mesh.guard[0] - 1
             ? 0
             : 1);
    size_t dx_minus =
        (boundary_lower && i == dev_mesh.guard[0] ? 0 : 1);
    j_tmp(i, 0) = 0.5f * j(i, 0);
    j_tmp(i, 0) += 0.25f * j(i + dx_plus, 0);
    j_tmp(i, 0) += 0.25f * j(i - dx_minus, 0);
  }
}

}  // namespace Kernels

ptc_updater_1dgap::ptc_updater_1dgap(sim_environment& env)
    : m_env(env) {
  m_tmp_j = multi_array<Scalar>(m_env.local_grid().extent());
}

ptc_updater_1dgap::~ptc_updater_1dgap() {}

void
ptc_updater_1dgap::update_particles(sim_data& data, double dt,
                                    uint32_t step) {
  // Grid_1dGR* grid = dynamic_cast<Grid_1dGR*>(&m_env.local_grid());
  // auto mesh_ptrs = grid->get_mesh_ptrs();
  auto data_p = get_data_ptrs(data);
  if (data.particles.number() > 0) {
    data.J.initialize();
    for (auto& rho : data.Rho) {
      rho.initialize();
    }

    // Filter E field before particle push
    data.E.data(2).copy_from(data.E.data(0));
    for (int i = 0; i < m_env.params().current_smoothing; i++) {
      Kernels::filter_current1d<<<256, 256>>>(
          get_pitchptr(data.E.data(2)), get_pitchptr(data.E.data(1)),
          m_env.is_boundary(0), m_env.is_boundary(1));
      CudaCheckError();
      data.E.data(2).copy_from(data.E.data(1));
    }

    Logger::print_info("Updating {} particles",
                       data.particles.number());
    Kernels::update_ptc_1dgap<<<
        std::min((data.particles.number() + 511) / 512, 2048ul), 512>>>(
        data_p, data.particles.number(), dt, step);
    CudaCheckError();

    Logger::print_info("smoothing current {} times",
                       m_env.params().current_smoothing);
    for (int i = 0; i < m_env.params().current_smoothing; i++) {
      Kernels::filter_current1d<<<256, 256>>>(
          get_pitchptr(data.J.data(0)), get_pitchptr(m_tmp_j),
          m_env.is_boundary(0), m_env.is_boundary(1));
      CudaCheckError();
      data.J.data(0).copy_from(m_tmp_j);

      for (int sp = 0; sp < m_env.params().num_species; sp++) {
        Kernels::filter_current1d<<<256, 256>>>(
            get_pitchptr(data.Rho[sp].data()), get_pitchptr(m_tmp_j),
            m_env.is_boundary(0), m_env.is_boundary(1));
        CudaCheckError();
        data.Rho[sp].data().copy_from(m_tmp_j);
      }
    }
  }

  if (data.photons.number() > 0) {
    Logger::print_info("Updating {} photons", data.photons.number());
    Kernels::update_photon_1dgap<<<
        std::min((data.photons.number() + 511) / 512, 2048ul), 512>>>(
        data.photons.data(), data.photons.number(), dt, step);
    CudaCheckError();
  }
}

void
ptc_updater_1dgap::prepare_initial_condition(sim_data& data,
                                             int multiplicity) {
  // const Grid_1dGap* g =
  //     dynamic_cast<const Grid_1dGap*>(&data.env.grid());
  auto g = m_env.local_grid();
  Kernels::prepare_initial_condition<<<128, 256>>>(
      data.particles.data(), m_env.params().B0 * m_env.params().omega,
      multiplicity);
  CudaCheckError();

  data.particles.set_num(g.mesh().reduced_dim(0) * 2 * multiplicity);
  // particles.append({0.5, 0.0, 4000})
}

}  // namespace Aperture
