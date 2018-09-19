#include "algorithms/ptc_pusher_geodesic.h"
#include "algorithms/functions.h"
#include "sim_environment.h"
#include "utils/logger.h"
#include "utils/util_functions.h"
#include <array>
#include <cmath>
#include <fmt/ostream.h>
#ifdef __AVX2__
#include <immintrin.h>
#include <xmmintrin.h>
#endif  // __AVX2__

namespace Aperture {

#ifdef __AVX2__

__m256d
beta_avx2(__m256d x) {
  __m256d result =
      _mm256_fmadd_pd(_mm256_sub_pd(x, _mm256_set1_pd(0.05)),
                      _mm256_set1_pd(1.0 / 0.45), _mm256_set1_pd(-1.0));
  return result;
}

__m256d
gamma_avx2(__m256d beta_phi, __m256d p) {
  __m256d b2p1 =
      _mm256_fmadd_pd(beta_phi, beta_phi, _mm256_set1_pd(1.0));
  __m256d gamma2 = _mm256_fmadd_pd(p, p, b2p1);
  return _mm256_sqrt_pd(gamma2);
}

__m256d
sign(__m256d x) {
  __m256d zero = _mm256_setzero_pd();

  __m256d positive = _mm256_and_pd(_mm256_cmp_pd(x, zero, _CMP_GT_OS),
                                   _mm256_set1_pd(1.0));
  __m256d negative = _mm256_and_pd(_mm256_cmp_pd(x, zero, _CMP_LT_OS),
                                   _mm256_set1_pd(-1.0));

  return _mm256_or_pd(positive, negative);
}

#endif  // __AVX2__

double
gamma(double beta_phi, double p) {
  double b2 = beta_phi * beta_phi;
  // if (beta_phi < 0) p = -p;

  // if (b2 > 1.0 && p*p/(1.0 + b2) + (1.0 - b2) < 0) {
  //   Logger::print_info("b2 is {}, p is {}, sqrt is {}, {}", b2, p,
  //   p*p/(1.0 + b2), (1.0 - b2));
  // }
  // double result = -p * b2 / std::sqrt(1.0 + b2) + std::sqrt(p*p/(1.0
  // + b2) + (1.0 - b2)); result *= 1.0 / (1.0 - b2);

  return std::sqrt(1.0 + p * p + b2);
}

ParticlePusher_Geodesic::ParticlePusher_Geodesic(
    const Environment& env) {}

ParticlePusher_Geodesic::~ParticlePusher_Geodesic() {}

void
ParticlePusher_Geodesic::push(SimData& data, double dt) {
  Logger::print_info("In particle pusher");
  auto& grid = data.E.grid();
  auto& mesh = grid.mesh();
  for (auto& particles : data.particles) {
#ifdef __AVX2_CUSTOM__
    auto& ptc = particles.data();
    for (Index_t idx = 0; idx + 3 < particles.number(); idx += 4) {
      // Lorentz push in avx2
      __m128i cell = _mm_load_si128((const __m128i*)(ptc.cell + idx));
      __m128i cell_m = _mm_sub_epi32(cell, _mm_set1_epi32(1));
      // Less than MAX_CELL means that the particle position is not
      // empty
      __m128i cell_mask =
          _mm_cmplt_epi32(cell, _mm_set1_epi32(MAX_CELL));
      // __m256i cell_mask = _mm256_castsi128_si256(cell_mask_128);

      __m256d x1 = _mm256_load_pd(ptc.x1 + idx);
      __m256d cx1 = _mm256_sub_pd(_mm256_set1_pd(1.0), x1);
      __m256d E_0 = _mm256_mask_i32gather_pd(
          _mm256_set1_pd(0.0), data.E.ptr(0), cell_m,
          _mm256_cvtepi32_pd(cell_mask), 1);
      __m256d E_1 = _mm256_mask_i32gather_pd(
          _mm256_set1_pd(0.0), data.E.ptr(0), cell,
          _mm256_cvtepi32_pd(cell_mask), 1);
      // __m256d E_1 = _mm256_i32gather_pd(data.E.ptr(0), cell, 1);
      __m256d vE = _mm256_fmadd_pd(E_0, cx1, _mm256_mul_pd(E_1, x1));

      __m256d p1 = _mm256_load_pd(ptc.p1 + idx);
      _mm256_store_pd(
          ptc.p1 + idx,
          _mm256_fmadd_pd(vE,
                          _mm256_set1_pd(particles.charge() * dt /
                                         particles.mass()),
                          p1));

      // TODO: Add centrifugal force term

      // Move particles in avx2
      // TODO: compute full x from the relative coordinate
      __m256d x;
      __m256d beta = beta_avx2(x);
      __m256d gamma = gamma_avx2(beta, p1);
      _mm256_store_pd(ptc.gamma, gamma);

      __m256d beta_sign = sign(beta);
      __m256d v = _mm256_fmadd_pd(
          beta, beta,
          _mm256_div_pd(_mm256_mul_pd(beta_sign, p1), gamma));
      v = _mm256_mul_pd(beta_sign, v);
      __m256d dx =
          _mm256_mul_pd(v, _mm256_set1_pd(dt / grid.mesh().delta[0]));
      _mm256_store_pd(ptc.x1 + idx, dx);
      x1 = _mm256_add_pd(x1, dx);
      __m256d delta_cell = _mm256_floor_pd(x1);
      cell = _mm_add_epi32(cell, _mm256_cvtpd_epi32(delta_cell));
      x1 = _mm256_sub_pd(x1, delta_cell);
      _mm_store_si128((__m128i*)(ptc.cell + idx), cell);
      _mm256_store_pd(ptc.x1 + idx, x1);
    }
    Index_t idx_start = (particles.number() / 4) * 4;
#else
    Index_t idx_start = 0;
#endif  // __AVX2__
    for (Index_t idx = idx_start; idx < particles.number(); idx++) {
      if (particles.is_empty(idx)) continue;
      auto& ptc = particles.data();
      auto c = mesh.get_cell_3d(ptc.cell[idx]);

      // Logger::print_info("Looping particle {}", idx);
      double x = grid.mesh().pos(0, c[0], ptc.x1[idx]);
      // Logger::print_info("Pushing particle at cell {} and position
      // {}",
      //                    ptc.cell[idx], x);

      lorentz_push(particles, idx, x, data.E, data.B, dt);
      // extra_force(particles, idx, x, grid, dt);
      move_ptc(particles, idx, x, grid, dt);
    }
  }
}

void
ParticlePusher_Geodesic::move_ptc(Particles& particles, Index_t idx,
                                  double x, const Grid& grid,
                                  double dt) {
  auto& ptc = particles.data();
  auto& mesh = grid.mesh();
  if (mesh.dim() == 1) {
    int cell = ptc.cell[idx];

    // ptc.gamma[idx] = sqrt(1.0 + ptc.p1[idx] * ptc.p1[idx]);
    double beta = beta_phi(x / mesh.sizes[0]);
    double g = gamma(beta, ptc.p1[idx]);
    // if (g < 1.0) g = 1.0;
    // if (std::abs(beta_phi(x/mesh.sizes[0])) > 1.0)
    //   // Logger::print_info("p is {}, beta is {}, g is {}, x is {}",
    //   ptc.p1[idx], beta_phi(x/mesh.sizes[0]), g, x);
    ptc.gamma[idx] = g;
    // double v = ptc.p1[idx] / ptc.gamma[idx];
    // Logger::print_info("Before move, v is {}, gamma is {}", v,
    // ptc.gamma[idx]);

    double v =
        ((beta < 0.0 ? -1.0 : 1.0) * ptc.p1[idx] / g + beta * beta) /
        (1.0 + beta * beta);
    if (beta < 0.0) {
      v *= -1.0;
    }
    ptc.dx1[idx] = v * dt / grid.mesh().delta[0];
    ptc.x1[idx] += ptc.dx1[idx];

    // Compute the change in particle cell
    // auto c = mesh.get_cell_3d(cell);
    int delta_cell = (int)std::floor(ptc.x1[idx]);
    // std::cout << delta_cell << std::endl;
    cell += delta_cell;
    // Logger::print_info("After move, c is {}, x1 is {}", c,
    // ptc.x1[idx]);

    ptc.cell[idx] = cell;
    // std::cout << ptc.x1[idx] << ", " << ptc.cell[idx] << std::endl;
    ptc.x1[idx] -= (Pos_t)delta_cell;
    // std::cout << ptc.x1[idx] << ", " << ptc.cell[idx] << std::endl;
  }
}

void
ParticlePusher_Geodesic::lorentz_push(Particles& particles, Index_t idx,
                                      double x,
                                      const VectorField<Scalar>& E,
                                      const VectorField<Scalar>& B,
                                      double dt) {
  auto& ptc = particles.data();
  if (E.grid().dim() == 1) {
    // Logger::print_debug("in lorentz, flag is {}", ptc.flag[idx]);
    if (!check_bit(ptc.flag[idx], ParticleFlag::ignore_EM)) {
      auto& mesh = E.grid().mesh();
      int cell = ptc.cell[idx];
      // Vec3<Pos_t> rel_x{ptc.x1[idx], 0.0, 0.0};
      auto rel_x = ptc.x1[idx];

      // Vec3<Scalar> vE = m_interp.interp_cell(ptc.x[idx].vec3(),
      // grid.); auto c = mesh.get_cell_3d(cell); std::cout << c <<
      // std::endl; Vec3<Scalar> vE = E.interpolate(c, rel_x, m_interp);
      Scalar vE =
          E.data(0)[cell] * rel_x + E.data(0)[cell - 1] * (1.0 - rel_x);
      // Logger::print_info("in lorentz, c = {}, E = {}, rel_x = {}", c,
      // vE, rel_x);

      double p = ptc.p1[idx];
      double beta = beta_phi(x / mesh.sizes[0]);
      double g = gamma(beta, p);
      double f =
          (g - (beta < 0.0 ? -1.0 : 1.0) * p) / (1.0 + beta * beta);
      ptc.p1[idx] += (beta / g) * f * f * dt / (0.5 * mesh.sizes[0]);
      ptc.p1[idx] += particles.charge() * vE * dt / particles.mass();
    }
  }
}

void
ParticlePusher_Geodesic::handle_boundary(SimData& data) {
  auto& mesh = data.E.grid().mesh();
  for (auto& ptc : data.particles) {
    if (ptc.number() > 0) {
      for (Index_t n = 0; n < ptc.number(); n++) {
        // This controls the boundary condition
        auto c = mesh.get_cell_3d(ptc.data().cell[n]);
        if (c[0] < mesh.guard[0] ||
            c[0] >= mesh.dims[0] - mesh.guard[0]) {
          // Move particles to the other end of the box
          if (m_periodic) {
            if (c[0] < mesh.guard[0])
              c[0] += mesh.reduced_dim(0);
            else
              c[0] -= mesh.reduced_dim(0);
            ptc.data().cell[n] = mesh.get_idx(c[0], c[1], c[2]);
          } else {
            // Erase particles in the guard cell
            if (c[0] <= 2 || c[0] >= mesh.dims[0] - 3) ptc.erase(n);
          }
        }
      }
    }
  }
  auto& ptc = data.photons;
  if (ptc.number() > 0) {
    for (Index_t n = 0; n < ptc.number(); n++) {
      // This controls the boundary condition
      auto c = mesh.get_cell_3d(ptc.data().cell[n]);
      if (c[0] < mesh.guard[0] ||
          c[0] >= mesh.dims[0] - mesh.guard[0]) {
        // Move particles to the other end of the box
        if (m_periodic) {
          if (c[0] < mesh.guard[0])
            c[0] += mesh.reduced_dim(0);
          else
            c[0] -= mesh.reduced_dim(0);
          ptc.data().cell[n] = mesh.get_idx(c[0], c[1], c[2]);
        } else {
          // Erase particles in the guard cell
          ptc.erase(n);
        }
      }
    }
  }
}

void
ParticlePusher_Geodesic::extra_force(Particles& particles, Index_t idx,
                                     double x, const Grid& grid,
                                     double dt) {
  auto& ptc = particles.data();

  auto& mesh = grid.mesh();

  // Add fake light surfaces
  // if (x < 0.1 * mesh.sizes[0] && ptc.p1[idx] > 0) {
  //   // repel like crazy
  //   ptc.p1[idx] = 0.0;
  // } else if (x > 0.9 * mesh.sizes[0] && ptc.p1[idx] < 0) {
  //   ptc.p1[idx] = 0.0;
  // }

  // double p = ptc.p1[idx] / 100.0;
  double g0 = 0.0;
  double f = (2.0 * x / mesh.sizes[0] - 1.3);
  double g = g0 * f;
  ptc.p1[idx] += g * particles.mass() * dt;

  // double drag = 0.5;

  // ptc.p1[idx] -= drag * p * p * p * dt;
}

}  // namespace Aperture
