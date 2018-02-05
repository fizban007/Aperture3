#include "algorithms/ptc_pusher_geodesic.h"
// #include "CudaLE.h"
#include "algorithms/solve.h"
#include "utils/util_functions.h"
// #include "metrics.h"
#include <array>
#include <cmath>
#include <fmt/ostream.h>

#define CONN(METRIC, I, A, B, X)                                               \
  D<I>(METRIC.inv_g##A##B + METRIC.b##A * METRIC.b##B / METRIC.a2)(X[0], X[1], \
                                                                   X[2])
// using CudaLE::D;

namespace Aperture {

// const int max_iter = 100;
// const double tolerance = 1.0e-14;
// const double x_diff = 1.0e-6;

ParticlePusher_Geodesic::ParticlePusher_Geodesic() {}

ParticlePusher_Geodesic::~ParticlePusher_Geodesic() {}

void
ParticlePusher_Geodesic::push(SimData& data, double dt) {
  auto& grid = data.E.grid();
  for (auto& particles : data.particles) {
    for (Index_t idx = 0; idx < particles.number(); idx++) {
      if (particles.is_empty(idx)) continue;
      auto& ptc = particles.data();
      // Vec3<double> x = grid.mesh().pos_particle(
      //     ptc.cell[idx], Vec3<float>{ptc.x1[idx], 0.0, 0.0});
      double x = grid.mesh().pos(0, ptc.cell[idx], ptc.x1[idx]);
      // Logger::print_debug_all("Pushing particle at cell {} and position {}",
      // ptc.cell[idx], x);

      lorentz_push(particles, idx, x, data.E, data.B, dt * 0.5);
      move(particles, idx, x, grid, dt);
      lorentz_push(particles, idx, x, data.E, data.B, dt * 0.5);
    }
  }
}

void
ParticlePusher_Geodesic::move(Particles& particles, Index_t idx,
                                       double x, const Grid& grid,
                                       double dt) {
  auto& ptc = particles.data();
  int cell = ptc.cell[idx];

  double v = ptc.p1[idx] / ptc.gamma[idx];
  ptc.dx1[idx] = v * dt;
  ptc.x1[idx] += v * dt;
  // Vec3<double> u = Vec3<double>{ptc.p1[idx], ptc.p2[idx], ptc.p3[idx]};
  // Logger::print_debug("Original u: {}", u);

  // Use rk4 to compute new position and momentum
  // iterate_rk4(x, u, grid, dt);
  // auto dx = x - x0;

  // Logger::print_debug("New u: {}", u);
  // update the position and momentum
  // ptc.p1[idx] = u[0];
  // ptc.p2[idx] = u[1];
  // ptc.p3[idx] = u[2];

  // ptc.dx1[idx] = dx[0] / grid.mesh().delta[0];
  // ptc.dx2[idx] = dx[1] / grid.mesh().delta[1];
  // ptc.dx3[idx] = dx[2];

  // ptc.x1[idx] += dx[0] / grid.mesh().delta[0];
  // ptc.x2[idx] += dx[1] / grid.mesh().delta[1];
  // ptc.x3[idx] += dx[2];

  // Logger::print_debug("Moving particle with dx {}", dx);
  // Logger::print_debug("Original x: {}", x0);
  // Logger::print_debug("New x: {}", x);
  // Logger::print_debug("After move, particle has x {}",
  //                     Vec3<float>(ptc.x1[idx], ptc.x2[idx], ptc.x3[idx]));

  // Compute the change in particle cell
  int delta_cell = (int)std::floor(ptc.x1[idx]);
  cell += delta_cell;
  ptc.x1[idx] -= (float)delta_cell;

}

void
ParticlePusher_Geodesic::lorentz_push(Particles& particles, Index_t idx,
                                      double x,
                                      const VectorField<Scalar>& E,
                                      const VectorField<Scalar>& B, double dt) {
  auto& ptc = particles.data();
  if (E.grid().dim() == 1) {
    // Logger::print_debug("in lorentz, flag is {}", ptc.flag[idx]);
    if (!check_bit(ptc.flag[idx], ParticleFlag::ignore_EM)) {
      auto& mesh = E.grid().mesh();
      int cell = ptc.cell[idx];
      Vec3<float> rel_x{ptc.x1[idx], 0.0, 0.0};

      // Vec3<Scalar> vE = m_interp.interp_cell(ptc.x[idx].vec3(), grid.);
      auto c = mesh.get_cell_3d(cell);
      Vec3<Scalar> vE = E.interpolate(c, rel_x, m_order);

      ptc.p1[idx] += particles.charge() * vE[0] * dt / particles.mass();

      // double gu1 = m_cache.invg13 * u[2] + m_cache.invg11 * u[0];
      // double gu2 = m_cache.invg22 * u[1];
      // double gu3 = m_cache.invg13 * u[0] + m_cache.invg33 * u[2];
      // u_prime[0] += coef * vE[0] + gu2 * vB[2] - gu3 * vB[1];
      // u_prime[1] += coef * vE[1] + gu3 * vB[0] - gu1 * vB[2];
      // u_prime[2] += coef * vE[2] + gu1 * vB[1] - gu2 * vB[0];

      // Logger::print_debug("u prime 2 is {}, u is {}", u_prime, u);

      // // Construct the 3x3 matrix
      // m_boris_mat[0][0] = 1.0 + m_cache.invg13 * vB[1];
      // m_boris_mat[0][1] = -m_cache.invg22 * vB[2];
      // m_boris_mat[1][0] = -(m_cache.invg13 * vB[0] - m_cache.invg11 * vB[2]);
      // m_boris_mat[0][2] = m_cache.invg33 * vB[1];
      // m_boris_mat[2][0] = -m_cache.invg11 * vB[1];
      // m_boris_mat[1][1] = 1.0;
      // m_boris_mat[1][2] = -(m_cache.invg33 * vB[0] - m_cache.invg13 * vB[2]);
      // m_boris_mat[2][1] = m_cache.invg22 * vB[0];
      // m_boris_mat[2][2] = 1.0 - m_cache.invg13 * vB[1];

      // // Solve the matrix and store the result
      // solve(m_boris_mat, u_prime, u);
      // ptc.p1[idx] = u[0];
      // ptc.p2[idx] = u[1];
      // ptc.p3[idx] = u[2];
    }
  }
}

}  // namespace Aperture
