#include "ptc_updater_default.h"
// #include "algorithms/interpolation.h"
#include "omp.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/avx_interp.hpp"
#include "utils/avx_utils.h"
#include "utils/simd.h"

namespace Aperture {

template <typename VF>
inline VF
center2d(const VF& sx0, const VF& sx1, const VF& sy0, const VF& sy1) {
  return mul_add(
             sx1, sy1 * 2.0,
             mul_add(sx0, sy1, mul_add(sx1, sy0, sx0 * sy0 * 2.0))) *
         0.1666667;
}

template <typename VF>
inline VF
center2ds(const VF& sx0, const VF& sx1, const VF& sy0, const VF& sy1) {
  return (sx1 * sy1 * 2.0 + sx0 * sy1 + sx1 * sy0 + sx0 * sy0 * 2.0) *
         0.1666667;
}

template <typename VF>
inline VF
movement3d(const VF& sx0, const VF& sx1, const VF& sy0, const VF& sy1,
           const VF& sz0, const VF& sz1) {
  return (sz1 - sz0) * center2d(sx0, sx1, sy0, sy1);
}

template <typename VF>
inline VF
movement3ds(const VF& sx0, const VF& sx1, const VF& sy0, const VF& sy1,
            const VF& sz0, const VF& sz1) {
  return (sz1 - sz0) * center2ds(sx0, sx1, sy0, sy1);
}

ptc_updater_default::ptc_updater_default(const sim_environment& env)
    : ptc_updater(env) {}
      // m_j1(env.grid().extent()),
      // m_j2(env.grid().extent()),
      // m_j3(env.grid().extent()) {}

ptc_updater_default::~ptc_updater_default() {}

void
ptc_updater_default::update_particles(sim_data& data, double dt, uint32_t step) {
  // m_j1.assign(simd_buffer{0.0});
  // m_j2.assign(simd_buffer{0.0});
  // m_j3.assign(simd_buffer{0.0});

  // TODO: Push photons as well
  push(data, dt);
  esirkepov_deposit(data, dt);
}

void
ptc_updater_default::push(sim_data& data, double dt, uint32_t step) {
  using namespace simd;

  auto& ptc = data.particles;
  auto& mesh = m_env.grid().mesh();
  if (ptc.number() > 0) {
    //    int vec_width = 8;
    Logger::print_info("vec_width is {}", vec_width);
    for (size_t idx = 0; idx < ptc.number(); idx += vec_width) {
      // Interpolate field values to particle position
      Vec_idx_type c;
      c.load_a(ptc.data().cell + idx);
      // Mask for empty particles. Do not write results to these
      // particles
      // Vec_ib_type empty_mask = (c != Vec_ui_type(MAX_CELL));
      auto d = c / mesh.dims[0];
      auto c1s = c - d * mesh.dims[0];
      uint32_t empty_offset =
          1 * sizeof(Scalar) +
          (1 + mesh.dims[1]) * data.E.data(0).pitch();
#ifdef USE_DOUBLE
      Vec_ui_type offsets =
          extend_low(c1s * sizeof(double) + d * data.E.data(0).pitch());
      Vec_ib_type empty_mask = (extend_low(c) != Vec_ui_type(MAX_CELL));
#else
      Vec_ui_type offsets =
          c1s * sizeof(float) + d * data.E.data(0).pitch();
      Vec_ib_type empty_mask = (c != Vec_ui_type(MAX_CELL));
#endif
      offsets = select(~empty_mask, Vec_ui_type(empty_offset), offsets);

      Vec_f_type x1;
      x1.maskload_a(ptc.data().x1 + idx, empty_mask);
      Vec_f_type x2;
      x2.maskload_a(ptc.data().x2 + idx, empty_mask);
      Vec_f_type x3;
      x3.maskload_a(ptc.data().x3 + idx, empty_mask);

      // Find q_over_m of the current particle species
      Vec_ui_type flag;
      flag.maskload_a((int*)(ptc.data().flag + idx), empty_mask);
      Vec_i_type sp = get_ptc_type(flag);
      Vec_f_type q_over_m = lookup<vec_width>(sp, m_env.q_over_m());

      Vec_f_type E1 = interpolate_3d(data.E.data(0), offsets, x1, x2,
                                     x3, data.E.stagger(0)) *
                      q_over_m * 2.0f;
      Vec_f_type E2 = interpolate_3d(data.E.data(1), offsets, x1, x2,
                                     x3, data.E.stagger(1)) *
                      q_over_m * 2.0f;
      Vec_f_type E3 = interpolate_3d(data.E.data(2), offsets, x1, x2,
                                     x3, data.E.stagger(2)) *
                      q_over_m * 2.0f;
      Vec_f_type B1 = interpolate_3d(data.B.data(0), offsets, x1, x2,
                                     x3, data.B.stagger(0)) *
                      q_over_m;
      Vec_f_type B2 = interpolate_3d(data.B.data(1), offsets, x1, x2,
                                     x3, data.B.stagger(1)) *
                      q_over_m;
      Vec_f_type B3 = interpolate_3d(data.B.data(2), offsets, x1, x2,
                                     x3, data.B.stagger(2)) *
                      q_over_m;

      Vec_f_type p1;
      p1.maskload_a(ptc.data().p1 + idx, empty_mask);
      Vec_f_type p2;
      p2.maskload_a(ptc.data().p2 + idx, empty_mask);
      Vec_f_type p3;
      p3.maskload_a(ptc.data().p3 + idx, empty_mask);
      Vec_f_type gamma;
      gamma.maskload_a(ptc.data().E + idx, empty_mask);

      // Vay push
      Vec_f_type up1 = p1 + E1 + mul_add(p2, B3, -p3 * B2) / gamma;
      Vec_f_type up2 = p2 + E2 + mul_add(p3, B1, -p1 * B3) / gamma;
      Vec_f_type up3 = p3 + E3 + mul_add(p1, B2, -p2 * B1) / gamma;

      Vec_f_type tt = mul_add(B1, B1, mul_add(B2, B2, B3 * B3));
      Vec_f_type ut = mul_add(up1, B1, mul_add(up2, B2, up3 * B3));

      Vec_f_type sigma = mul_add(
          up1, up1,
          mul_add(up2, up2, mul_add(up3, up3, Vec_f_type(1.0f) - tt)));
      Vec_f_type inv_gamma2 =
          Vec_f_type(2.0f) /
          (sigma +
           sqrt(mul_add(sigma, sigma, mul_add(ut, ut, tt) * 4.0f)));
      Vec_f_type s =
          Vec_f_type(1.0f) / mul_add(inv_gamma2, tt, Vec_f_type(1.0f));
      Vec_f_type inv_gamma = sqrt(inv_gamma2);
      gamma = Vec_f_type(1.0f) / inv_gamma;

      gamma.maskstore_a(ptc.data().E + idx, empty_mask);

      p1 = s * mul_add(mul_add(up2, B3, -up3 * B2), inv_gamma,
                       mul_add(B1 * ut, inv_gamma2, up1));
      p2 = s * mul_add(mul_add(up3, B1, -up1 * B3), inv_gamma,
                       mul_add(B2 * ut, inv_gamma2, up2));
      p3 = s * mul_add(mul_add(up1, B2, -up2 * B1), inv_gamma,
                       mul_add(B3 * ut, inv_gamma2, up3));
      // p2 = s * (mul_add(B2 * ut, inv_gamma2, up2) +
      //           mul_add(up3, B1, -up1 * B3) * inv_gamma);
      // p3 = s * (mul_add(B3 * ut, inv_gamma2, up3) +
      //           mul_add(up1, B2, -up2 * B1) * inv_gamma);
      // Vec_f_type pm1 = p1 + E1;
      // Vec_f_type pm2 = p2 + E2;
      // Vec_f_type pm3 = p3 + E3;
      // Vec_f_type gamma = sqrt(pm1 * pm1 + pm2 * pm2 + pm3 * pm3
      // + 1.0f);

      // gamma.store_a(ptc.data().E + idx);

      // Vec_f_type pp1 = pm1 + (pm2 * B3 - pm3 * B2) / gamma;
      // Vec_f_type pp2 = pm2 + (pm3 * B1 - pm1 * B3) / gamma;
      // Vec_f_type pp3 = pm3 + (pm1 * B2 - pm2 * B1) / gamma;
      // Vec_f_type t2p1 =
      //     (B1 * B1 + B2 * B2 + B3 * B3) / (gamma * gamma) + 1.0f;

      // p1 = E1 + pm1 + (pp2 * B3 - pp3 * B2) / t2p1 * 2.0f;
      // p2 = E2 + pm2 + (pp3 * B1 - pp1 * B3) / t2p1 * 2.0f;
      // p3 = E3 + pm3 + (pp1 * B2 - pp2 * B1) / t2p1 * 2.0f;

      p1.maskstore_a(ptc.data().p1 + idx, empty_mask);
      p2.maskstore_a(ptc.data().p2 + idx, empty_mask);
      p3.maskstore_a(ptc.data().p3 + idx, empty_mask);
    }
  }
}

void
ptc_updater_default::esirkepov_deposit(sim_data& data, double dt, uint32_t step) {
  auto& ptc = data.particles;
  auto& mesh = m_env.grid().mesh();
  if (ptc.number() > 0) {
#pragma omp simd
    for (size_t idx = 0; idx < ptc.number(); idx++) {
      uint32_t c = ptc.data().cell[idx];
      if (c == MAX_CELL) continue;

      Pos_t x1 = ptc.data().x1[idx];
      Pos_t x2 = ptc.data().x2[idx];
      Pos_t x3 = ptc.data().x3[idx];
      Scalar p1 = ptc.data().p1[idx];
      Scalar p2 = ptc.data().p2[idx];
      Scalar p3 = ptc.data().p3[idx];
      Scalar gamma = ptc.data().E[idx];

      // Move particles
      p1 /= gamma;
      p2 /= gamma;
      p3 /= gamma;

      Pos_t new_x1 = x1 + p1 * dt * mesh.inv_delta[0];
      Pos_t new_x2 = x2 + p2 * dt * mesh.inv_delta[1];
      Pos_t new_x3 = x3 + p3 * dt * mesh.inv_delta[2];

      // Deposit current
      uint32_t flag = ptc.data().flag[idx];
      int sp = get_ptc_type(flag);
      Scalar weight = ptc.data().weight[idx];
      weight *= -m_env.charge(sp);

      size_t offset = (c % mesh.dims[0]) * sizeof(Scalar) +
                      (c / mesh.dims[0]) * data.J.data(0).pitch();

      int k_0 = (new_x3 < 0.0 ? -2 : -1);
      int k_1 = (new_x3 > 1.0 ? 1 : 0);
      int j_0 = (new_x2 < 0.0 ? -2 : -1);
      int j_1 = (new_x2 > 1.0 ? 1 : 0);
      int i_0 = (new_x1 < 0.0 ? -2 : -1);
      int i_1 = (new_x1 > 1.0 ? 1 : 0);
      Scalar djz[3 * 3] = {0.0};
      for (int k = k_0; k <= k_1; k++) {
        Scalar sz0 = interp_1(-x3 + (k + 1));
        Scalar sz1 = interp_1(-new_x3 + (k + 1));
        int k_offset = k * mesh.dims[1];

        Scalar djy[3] = {0.0};
        for (int j = j_0; j <= j_1; j++) {
          Scalar sy0 = interp_1(-x2 + (j + 1));
          Scalar sy1 = interp_1(-new_x2 + (j + 1));
          int j_offset = (j + k_offset) * data.J.data(0).pitch();

          Scalar djx = 0.0f;
          for (int i = i_0; i <= i_1; i++) {
            Scalar sx0 = interp_1(-x1 + (i + 1));
            Scalar sx1 = interp_1(-new_x1 + (i + 1));

            int ij = i - i_0 + 3 * (j - j_0);

            djx += movement3ds(sx0, sx1, sy0, sy1, sz0, sz1);
            djy[i - i_0] += movement3ds(sy0, sy1, sz0, sz1, sx0, sx1);
            djz[ij] += movement3ds(sz0, sz1, sx0, sx1, sy0, sy1);

            Scalar s1 = sx1 * sy1 * sz1;
            size_t off = offset + i * sizeof(Scalar) + j_offset;

            data.J.data(0)[off + sizeof(Scalar)] += weight * djx;
            data.J.data(1)[off + data.J.data(1).pitch()] +=
                weight * djy[i - i_0];
            data.J.data(
                2)[off + mesh.dims[1] * data.J.data(1).pitch()] +=
                weight * djz[ij];
            data.Rho[sp].data()[off] -= weight * s1;
          }
        }
      }

      // Move the particles
      auto dc1 = floor(new_x1);
      new_x1 -= dc1;
      auto dc2 = floor(new_x2);
      new_x2 -= dc2;
      auto dc3 = floor(new_x3);
      new_x3 -= dc3;

      ptc.data().cell[idx] = c + int(dc1) + int(dc2) * mesh.dims[0] +
                             int(dc3) * (mesh.dims[0] * mesh.dims[1]);
      ptc.data().x1[idx] = new_x1;
      ptc.data().x2[idx] = new_x2;
      ptc.data().x3[idx] = new_x3;
    }
  }
}

void
ptc_updater_default::update_particles_slow(sim_data& data, double dt) {
  auto& ptc = data.particles;
#pragma omp simd
  for (Index_t idx = 0; idx < ptc.number(); idx++) {
    // Interpolate field values to particle position
    uint32_t c = ptc.data().cell[idx];
    Scalar x1 = ptc.data().x1[idx];
    Scalar x2 = ptc.data().x2[idx];
    Scalar x3 = ptc.data().x3[idx];

    // float q_over_m = dt * 0.5f;

    Scalar E1 =
        data.E.data(0).interpolate(c, x1, x2, x3, data.E.stagger(0));
    Scalar E2 =
        data.E.data(1).interpolate(c, x1, x2, x3, data.E.stagger(1));
    Scalar E3 =
        data.E.data(2).interpolate(c, x1, x2, x3, data.E.stagger(2));
    Scalar B1 =
        data.B.data(0).interpolate(c, x1, x2, x3, data.B.stagger(0));
    Scalar B2 =
        data.B.data(1).interpolate(c, x1, x2, x3, data.B.stagger(1));
    Scalar B3 =
        data.B.data(2).interpolate(c, x1, x2, x3, data.B.stagger(2));

    Scalar p1 = ptc.data().p1[idx];
    Scalar p2 = ptc.data().p2[idx];
    Scalar p3 = ptc.data().p3[idx];

    auto pm1 = p1 + E1;
    auto pm2 = p2 + E2;
    auto pm3 = p3 + E3;
    auto gamma = sqrt(pm1 * pm1 + pm2 * pm2 + pm3 * pm3 + 1.0f);

    // gamma.store_a(ptc.data().E + idx);
    ptc.data().E[idx] = gamma;

    auto pp1 = pm1 + (pm2 * B3 - pm3 * B2) / gamma;
    auto pp2 = pm2 + (pm3 * B1 - pm1 * B3) / gamma;
    auto pp3 = pm3 + (pm1 * B2 - pm2 * B1) / gamma;
    auto t2p1 = (B1 * B1 + B2 * B2 + B3 * B3) / (gamma * gamma) + 1.0f;

    p1 = E1 + pm1 + (pp2 * B3 - pp3 * B2) / t2p1 * 2.0f;
    p2 = E2 + pm2 + (pp3 * B1 - pp1 * B3) / t2p1 * 2.0f;
    p3 = E3 + pm3 + (pp1 * B2 - pp2 * B1) / t2p1 * 2.0f;

    // p1.store_a(ptc.data().p1 + idx);
    // p2.store_a(ptc.data().p2 + idx);
    // p3.store_a(ptc.data().p3 + idx);
    ptc.data().p1[idx] = p1;
    ptc.data().p2[idx] = p2;
    ptc.data().p3[idx] = p3;
  }
}

void
ptc_updater_default::handle_boundary(sim_data& data) {}

}  // namespace Aperture
