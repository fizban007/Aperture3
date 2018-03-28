#include "algorithms/ptc_pusher_geodesic.h"
#include "utils/util_functions.h"
#include <array>
#include <cmath>
#include <fmt/ostream.h>
#include "utils/logger.h"

namespace Aperture {

double gamma(double beta_phi, double p) {
  double b2 = beta_phi * beta_phi;
  // if (beta_phi < 0) p = -p;

  // if (b2 > 1.0 && p*p/(1.0 + b2) + (1.0 - b2) < 0) {
  //   Logger::print_info("b2 is {}, p is {}, sqrt is {}, {}", b2, p, p*p/(1.0 + b2), (1.0 - b2));
  // }
  // double result = -p * b2 / std::sqrt(1.0 + b2) + std::sqrt(p*p/(1.0 + b2) + (1.0 - b2));
  // result *= 1.0 / (1.0 - b2);

  return std::sqrt(1.0 + p*p + b2);
}

double beta_phi(double x) {
  double b = (x - 0.1)/0.4 - 1.0;
  if (x > 0.1 && x < 0.9) {
    return b;
  } else {
    return b;
  }
}

ParticlePusher_Geodesic::ParticlePusher_Geodesic() {}

ParticlePusher_Geodesic::~ParticlePusher_Geodesic() {}

void
ParticlePusher_Geodesic::push(SimData& data, double dt) {
  Logger::print_info("In particle pusher");
  auto& grid = data.E.grid();
  auto& mesh = grid.mesh();
  for (auto& particles : data.particles) {
    for (Index_t idx = 0; idx < particles.number(); idx++) {
      if (particles.is_empty(idx)) continue;
      auto& ptc = particles.data();
      auto c = mesh.get_cell_3d(ptc.cell[idx]);

      // Logger::print_info("Looping particle {}", idx);
      double x = grid.mesh().pos(0, c[0], ptc.x1[idx]);
      // Logger::print_info("Pushing particle at cell {} and position {}",
      //                    ptc.cell[idx], x);

      lorentz_push(particles, idx, x, data.E, data.B, dt);
      extra_force(particles, idx, x, grid, dt);
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
  int cell = ptc.cell[idx];

  // ptc.gamma[idx] = sqrt(1.0 + ptc.p1[idx] * ptc.p1[idx]);
  double beta = beta_phi(x/mesh.sizes[0]);
  double g = gamma(beta, ptc.p1[idx]);
  // if (g < 1.0) g = 1.0;
  // if (std::abs(beta_phi(x/mesh.sizes[0])) > 1.0)
  //   // Logger::print_info("p is {}, beta is {}, g is {}, x is {}", ptc.p1[idx], beta_phi(x/mesh.sizes[0]), g, x);
  ptc.gamma[idx] = g;
  // double v = ptc.p1[idx] / ptc.gamma[idx];
  // Logger::print_info("Before move, v is {}, gamma is {}", v, ptc.gamma[idx]);

  double v = ((beta < 0.0 ? -1.0 : 1.0) * ptc.p1[idx] / g + beta * beta) / (1.0 + beta * beta);
  if (beta < 0.0) {
    v *= -1.0;
  }
  ptc.dx1[idx] = v * dt / grid.mesh().delta[0];
  ptc.x1[idx] += ptc.dx1[idx];

  // Compute the change in particle cell
  auto c = mesh.get_cell_3d(cell);
  int delta_cell = (int)std::floor(ptc.x1[idx]);
  // std::cout << delta_cell << std::endl;
  c[0] += delta_cell;
  // Logger::print_info("After move, c is {}, x1 is {}", c, ptc.x1[idx]);

  ptc.cell[idx] = mesh.get_idx(c[0], c[1], c[2]);
  // std::cout << ptc.x1[idx] << ", " << ptc.cell[idx] << std::endl;
  ptc.x1[idx] -= (Pos_t)delta_cell;
  // std::cout << ptc.x1[idx] << ", " << ptc.cell[idx] << std::endl;
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
      Vec3<Pos_t> rel_x{ptc.x1[idx], 0.0, 0.0};

      // Vec3<Scalar> vE = m_interp.interp_cell(ptc.x[idx].vec3(), grid.);
      auto c = mesh.get_cell_3d(cell);
      // std::cout << c << std::endl;
      Vec3<Scalar> vE = E.interpolate(c, rel_x, m_interp);
      // Logger::print_info("in lorentz, c = {}, E = {}, rel_x = {}", c, vE, rel_x);

      double p = ptc.p1[idx];
      double beta = beta_phi(x/mesh.sizes[0]);
      double g = gamma(beta, p);
      double f = (g - (beta < 0.0 ? -1.0 : 1.0) * p) / (1.0 + beta * beta);
      ptc.p1[idx] += (beta / g) * f * f * dt / (0.5 * mesh.sizes[0]);
      ptc.p1[idx] += particles.charge() * vE[0] * dt / particles.mass();

      // double b = beta_phi(x/mesh.sizes[0]);
      // double lim = std::sqrt((b*b - 1.0)*(b*b + 1.0));
      // if (b > 1.0 && ptc.p1[idx] < lim) {
      //   ptc.p1[idx] = lim + 1e-5;
      //   // Logger::print_info("lim is {}", lim);
      // }

      // if (b < -1.0 && ptc.p1[idx] > -lim)
      //   ptc.p1[idx] = -lim - 1e-5;
      // // ptc.gamma[idx] = sqrt(1.0 + ptc.p1[idx] * ptc.p1[idx]);

    }
  }
}

void
ParticlePusher_Geodesic::handle_boundary(SimData &data) {
  auto& mesh = data.E.grid().mesh();
  for (auto& ptc : data.particles) {
    if (ptc.number() > 0) {
      for (Index_t n = 0; n < ptc.number(); n++) {
        // This controls the boundary condition
        auto c = mesh.get_cell_3d(ptc.data().cell[n]);
        if (c[0] < mesh.guard[0] || c[0] >= mesh.dims[0] - mesh.guard[0]) {
          // Move particles to the other end of the box
          if (m_periodic) {
            if (c[0] < mesh.guard[0])
              c[0] += mesh.reduced_dim(0);
            else
              c[0] -= mesh.reduced_dim(0);
            ptc.data().cell[n] = mesh.get_idx(c[0], c[1], c[2]);
          } else {
            // Erase particles in the guard cell
            if (c[0] <= 2 || c[0] >= mesh.dims[0] - 3)
            ptc.erase(n);
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
      if (c[0] < mesh.guard[0] || c[0] >= mesh.dims[0] - mesh.guard[0]) {
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
ParticlePusher_Geodesic::extra_force(Particles &particles, Index_t idx, double x, const Grid &grid, double dt) {
  auto& ptc = particles.data();

  auto& mesh = grid.mesh();

  // Add fake light surfaces
  // if (x < 0.1 * mesh.sizes[0] && ptc.p1[idx] > 0) {
  //   // repel like crazy
  //   ptc.p1[idx] = 0.0;
  // } else if (x > 0.9 * mesh.sizes[0] && ptc.p1[idx] < 0) {
  //   ptc.p1[idx] = 0.0;
  // }

  double p = ptc.p1[idx] / 100.0;
  double g0 = 0.0;
  double f = (2.0 * x / mesh.sizes[0] - 1.3);
  double g = g0 * f;
  ptc.p1[idx] += g * particles.mass() * dt;

  // double drag = 0.5;

  // ptc.p1[idx] -= drag * p * p * p * dt;
}

}  // namespace Aperture
