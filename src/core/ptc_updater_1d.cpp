#include "ptc_updater_1d.h"
#include "omp.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/avx_interp.hpp"
#include "utils/avx_utils.h"
#include "utils/simd.h"
#include "utils/logger.h"

namespace Aperture {

ptc_updater_1d::ptc_updater_1d(const sim_environment& env)
    : ptc_updater(env) {}

ptc_updater_1d::~ptc_updater_1d() {}

void
ptc_updater_1d::update_particles(sim_data& data, double dt,
                                 uint32_t step) {
  push(data, dt, step);
  deposit(data, dt, step);
  data.particles.clear_guard_cells(data.E.grid());
}

void
ptc_updater_1d::push(sim_data& data, double dt, uint32_t step) {
  using namespace simd;

  auto& ptc = data.particles;
  auto& mesh = m_env.grid().mesh();
  if (mesh.dim() != 1) {
    Logger::print_info("Grid is not 1d, doing nothing in push");
    return;
  }
  if (ptc.number() > 0) {
    for (size_t idx = 0; idx < ptc.number(); idx += vec_width) {
      // Interpolate field values to particle position
      Vec_idx_type c;
      c.load_a(ptc.data().cell + idx);
      uint32_t empty_offset = sizeof(Scalar);
#ifdef USE_DOUBLE
      Vec_ui_type offsets = extend_low(c * sizeof(double));
      Vec_ib_type empty_mask = (extend_low(c) != Vec_ui_type(MAX_CELL));
#else
      Vec_ui_type offsets = c * sizeof(float);
      Vec_ib_type empty_mask = (c != Vec_ui_type(MAX_CELL));
#endif
      offsets = select(~empty_mask, Vec_ui_type(empty_offset), offsets);

#ifndef USE_DOUBLE
      Vec_f_type x1;
      x1.maskload_a(ptc.data().x1 + idx, empty_mask);

      // Find q_over_m of the current particle species
      Vec_ui_type flag;
      flag.maskload_a((int*)(ptc.data().flag + idx), empty_mask);
      Vec_i_type sp = get_ptc_type(flag);
      Vec_f_type q_over_m = lookup<vec_width>(sp, m_env.q_over_m());

      Vec_f_type E0 = gather((Scalar*)data.E.data(0).data(),
                             offsets - sizeof(Scalar), 1);
      Vec_f_type E1 =
          gather((Scalar*)data.E.data(0).data(), offsets, 1);

      Vec_f_type E = lerp(x1, E0, E1);

      Vec_f_type p1;
      p1.maskload_a(ptc.data().p1 + idx, empty_mask);

      p1 += E * q_over_m * dt;

      p1.maskstore_a(ptc.data().p1 + idx, empty_mask);

      auto gamma = sqrt(mul_add(p1, p1, Vec_f_type(1.0)));

      gamma.maskstore_a(ptc.data().E + idx, empty_mask);
#endif
    }
  }
}

void
ptc_updater_1d::deposit(sim_data& data, double dt, uint32_t step) {
  auto& ptc = data.particles;
  auto& mesh = m_env.grid().mesh();
  if (mesh.dim() != 1) {
    Logger::print_info("Grid is not 1d, doing nothing in deposit");
    return;
  }
  data.J.initialize();
  if ((step + 1) % m_env.params().data_interval == 0) {
    for (int sp = 0; sp < m_env.params().num_species; sp++)
      data.Rho[sp].initialize();
  }

  if (ptc.number() > 0) {
#pragma omp simd
    for (size_t idx = 0; idx < ptc.number(); idx++) {
      uint32_t c = ptc.data().cell[idx];
      if (c == MAX_CELL) continue;

      Pos_t x1 = ptc.data().x1[idx];
      Scalar p1 = ptc.data().p1[idx];
      Scalar gamma = ptc.data().E[idx];

      p1 /= gamma;

      Pos_t new_x1 = x1 + p1 * dt * mesh.inv_delta[0];

      // Deposit current
      uint32_t flag = ptc.data().flag[idx];
      int sp = get_ptc_type(flag);
      Scalar weight = ptc.data().weight[idx];
      weight *= -m_env.charge(sp);

      size_t offset = c * sizeof(Scalar);

      int i_0 = (new_x1 < 0.0 ? -2 : -1);
      int i_1 = (new_x1 > 1.0 ? 1 : 0);
      Scalar djx = 0.0;
      for (int i = i_0; i <= i_1; i++) {
        Scalar sx0 = interp_1(-x1 + (i + 1));
        Scalar sx1 = interp_1(-new_x1 + (i + 1));

        djx += sx1 - sx0;

        size_t off = offset + i * sizeof(Scalar);

        data.J.data(0)[off + sizeof(Scalar)] +=
            weight * djx * mesh.delta[0] / dt;

        if ((step + 1) % m_env.params().data_interval == 0) {
          data.Rho[sp].data()[off] -= weight * sx1;
        }
      }

      auto dc1 = floor(new_x1);
      new_x1 -= dc1;
      ptc.data().cell[idx] = c + int(dc1);
      ptc.data().x1[idx] = new_x1;
    }
  }
}

void
ptc_updater_1d::handle_boundary(sim_data& data) {
  data.particles.clear_guard_cells(data.E.grid());
}

}  // namespace Aperture
