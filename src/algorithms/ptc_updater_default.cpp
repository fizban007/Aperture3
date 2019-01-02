#include "ptc_updater_default.h"
// #include "algorithms/interpolation.h"
#include "algorithms/ptc_updater_avx_helper.h"
#include "omp.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/avx_interp.hpp"
#include "utils/avx_utils.h"
#include "utils/simd.h"

namespace Aperture {

ptc_updater_default::ptc_updater_default(const sim_environment &env)
    : ptc_updater(env),
      m_j1(env.grid().extent()),
      m_j2(env.grid().extent()),
      m_j3(env.grid().extent()) {}

ptc_updater_default::~ptc_updater_default() {}

void
ptc_updater_default::update_particles(sim_data &data, double dt) {
  using namespace simd;
  // m_j1.assign(simd_buffer{0.0});
  // m_j2.assign(simd_buffer{0.0});
  // m_j3.assign(simd_buffer{0.0});

  // TODO: Push photons as well
  auto &ptc = data.particles;
  auto &mesh = m_env.grid().mesh();
  if (ptc.number() > 0) {
    //    int vec_width = 8;
    Logger::print_info("vec_width is {}", vec_width);
    for (size_t idx = 0; idx < ptc.number(); idx += vec_width) {
      // Interpolate field values to particle position
      Vec_ui_type c;
      c.load_a(ptc.data().cell + idx);
      // Mask for empty particles. Do not write results to these
      // particles
      Vec_ib_type empty_mask = (c != Vec_ui_type(MAX_CELL));
      Vec_ui_type d = select(~empty_mask, Vec_ui_type(1 + mesh.dims[1]),
                             c / Divisor_ui(mesh.dims[0]));
      Vec_ui_type c1s =
          select(~empty_mask, Vec_ui_type(1), c - d * mesh.dims[0]);
      Vec_ui_type offsets =
          c1s * sizeof(float) + d * data.E.data(0).pitch();

      Vec_f_type x1;
      x1.maskload_a(ptc.data().x1 + idx, empty_mask);
      Vec_f_type x2;
      x2.maskload_a(ptc.data().x2 + idx, empty_mask);
      Vec_f_type x3;
      x3.maskload_a(ptc.data().x3 + idx, empty_mask);

      // Find q_over_m of the current particle species
      Vec_ui_type flag;
      flag.maskload_a((int *)(ptc.data().flag + idx), empty_mask);
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

      // Move particles
      p1 *= inv_gamma;
      p2 *= inv_gamma;
      p3 *= inv_gamma;
      Vec_f_type new_x1 = x1 + p1 * dt * mesh.inv_delta[0];
      Vec_f_type new_x2 = x2 + p2 * dt * mesh.inv_delta[1];
      Vec_f_type new_x3 = x3 + p3 * dt * mesh.inv_delta[2];

      // new_x1.maskstore_a(ptc.data().x1 + idx, empty_mask);
      // new_x2.maskstore_a(ptc.data().x2 + idx, empty_mask);
      // new_x3.maskstore_a(ptc.data().x3 + idx, empty_mask);

      // Deposit current
      Vec_f_type weight;
      weight.maskload_a(ptc.data().weight + idx, empty_mask);
      weight *= -lookup<vec_width>(sp, m_env.charges());

      Vec_f_type k_0 =
          select(new_x3 < 0.0, Vec_f_type(-2.0), Vec_f_type(-1.0));
      Vec_f_type j_0 =
          select(new_x2 < 0.0, Vec_f_type(-2.0), Vec_f_type(-1.0));
      Vec_f_type i_0 =
          select(new_x1 < 0.0, Vec_f_type(-2.0), Vec_f_type(-1.0));

      Vec_f_type djz[3 * 3] = {Vec_f_type(0.0f)};
      for (int k = 0; k < 3; k++) {
        auto sz0 = interp_1(-x3 + k_0 + float(k + 1));
        auto sz1 = interp_1(-new_x3 + k_0 + float(k + 1));
        int k_offset = k * mesh.dims[1];

        Vec_f_type djy[3] = {Vec_f_type(0.0f)};
        for (int j = 0; j < 3; j++) {
          auto sy0 = interp_1(-x2 + j_0 + float(j + 1));
          auto sy1 = interp_1(-new_x2 + j_0 + float(j + 1));
          size_t j_offset = (j + k_offset) * data.J.data(0).pitch();

          Vec_f_type djx(0.0f);
          for (int i = 0; i < 3; i++) {
            auto sx0 = interp_1(-x1 + i_0 + float(i + 1));
            auto sx1 = interp_1(-new_x1 + i_0 + float(i + 1));

            Vec_ui_type off = offsets + (i * sizeof(Scalar) + j_offset);
            // off += Vec8i(0, 1, 2, 3, 4, 5, 6, 7) * sizeof(Scalar);

            int ij = i + 3 * j;
            djx +=
                select(empty_mask,
                       movement3d(sx0, sx1, sy0, sy1, sz0, sz1), 0.0f);
            // Vec_f_type j1 = gather((float *)m_j1.data(), off, 1);
            // j1 += djx;
            // scatter(off, j1, (char *)m_j1.data());

            djy[i] +=
                select(empty_mask,
                       movement3d(sy0, sy1, sz0, sz1, sx0, sx1), 0.0f);
            // Vec_f_type j2 = gather((float *)m_j2.data(), off, 1);
            // j2 += djy[i];
            // scatter(off, j2, (char *)m_j2.data());

            djz[ij] +=
                select(empty_mask,
                       movement3d(sz0, sz1, sx0, sx1, sy0, sy1), 0.0f);
            // Vec_f_type j3 = gather((float *)m_j3.data(), off, 1);
            // j3 += djz[ij];
            // scatter(off, j3, (char *)m_j3.data());

            Vec_f_type s1 = sx1 * sy1 * sz1;

            // for (int n = 0; n < vec_width; n++) {
            //   auto target = off[n];
            //   if (target == (uint32_t)-1) continue;
            //   Vec_ib_type same_as_target = (off == target);
            //   off = select(same_as_target, (uint32_t)-1, off);

            //   Vec_f_type vals = select(same_as_target, djx, 0.0f);
            //   auto result = horizontal_add(vals);
            //   data.J.data(0)[target] += result;

            //   vals = select(same_as_target, djy[i], 0.0f);
            //   result = horizontal_add(vals);
            //   data.J.data(1)[target] += result;

            //   vals = select(same_as_target, djz[ij], 0.0f);
            //   result = horizontal_add(vals);
            //   data.J.data(2)[target] += result;
            // }
            // Vec_f_type j1 =
            //     gather((float *)data.J.data(0).data(), off, 1);
            // Vec_f_type j2 =
            //     gather((float *)data.J.data(1).data(), off, 1);
            // Vec_f_type j3 =
            //     gather((float *)data.J.data(2).data(), off, 1);
            // j1 += djx;
            // j2 += djy[i];
            // j3 += djz[ij];
            // scatter(off, j1, (char *)data.J.data(0).data());
            // scatter(off, j2, (char *)data.J.data(1).data());
            // scatter(off, j3, (char *)data.J.data(2).data());

            // Vec_f_type rho;
            // for (int n = 0; n < m_env.params().num_species; n++) {
            //   rho = gather((float *)data.Rho[n].data().data(), off,
            //   1); rho = if_add(sp == n, rho, s1); scatter(off, rho,
            //   (char *)data.Rho[n].data().data());
            // }
#pragma omp simd
            for (int n = 0; n < vec_width; n++) {
              size_t offset = off[n];
              data.J.data(0)[offset] += djx[n];
              data.J.data(1)[offset] += djy[i][n];
              data.J.data(2)[offset] += djz[ij][n];
              if (sp[n] == 0)
                data.Rho[0].data()[offset] += s1[n];
              else if (sp[n] == 1)
                data.Rho[1].data()[offset] += s1[n];
              else if (sp[n] == 2)
                data.Rho[2].data()[offset] += s1[n];
            }
          }
        }
      }
  // }

  Vec_i_type dc1 = round_to_int(floor(new_x1));
  Vec_i_type dc2 = round_to_int(floor(new_x2));
  Vec_i_type dc3 = round_to_int(floor(new_x3));
  new_x1 -= to_float(dc1);
  new_x2 -= to_float(dc2);
  new_x3 -= to_float(dc3);

  Vec_ui_type new_c =
      Vec_ui_type(Vec_i_type(c) + dc1 + dc2 * mesh.dims[0] +
                  dc3 * (mesh.dims[0] * mesh.dims[1]));
  new_c.maskstore_a((int *)(ptc.data().cell + idx), empty_mask);

  new_x1.maskstore_a(ptc.data().x1 + idx, empty_mask);
  new_x2.maskstore_a(ptc.data().x2 + idx, empty_mask);
  new_x3.maskstore_a(ptc.data().x3 + idx, empty_mask);
}

  }
}
// #else
//     for (size_t idx = 0; idx < ptc.number(); idx++) {
//     }
// #endif
// }

// void
// ptc_updater_default::vay_push(sim_data &data, double dt) {
// }

// void
// ptc_updater_default::esirkepov_deposit(sim_data &data, double dt) {
//   auto &ptc = data.particles;
//   if (ptc.number() > 0) {

//   }
// }

void
ptc_updater_default::update_particles_slow(sim_data &data, double dt) {
  auto &ptc = data.particles;
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
ptc_updater_default::handle_boundary(sim_data &data) {}

}  // namespace Aperture
