#include "data/photons.h"
#include "sim_environment.h"
#include "utils/logger.h"
#include "utils/util_functions.h"

namespace Aperture {

Photons::Photons() {}

Photons::Photons(std::size_t max_num)
    : ParticleBase<single_photon_t>(max_num),
    m_dist(0.0, 1.0) {
  // No environment provided, all pair creation parameters going to be default
}

Photons::Photons(const Environment& env)
    : ParticleBase<single_photon_t>((std::size_t)env.conf().max_photon_number),
    m_dist(0.0, 1.0) {
  create_pairs = env.conf().create_pairs;
  trace_photons = env.conf().trace_photons;
  gamma_thr = env.conf().gamma_thr;
  l_ph = env.conf().photon_path;
  p_ph = env.conf().delta_t / l_ph;
  p_ic = env.conf().delta_t / env.conf().ic_path;
  track_pct = env.conf().track_percent;

  alpha = env.conf().spectral_alpha;
  e_s = env.conf().e_s;
  e_min = env.conf().e_min;
  Logger::print_info("Photon conversion probability is {}", p_ph);
}

Photons::Photons(const Photons& other)
    : ParticleBase<single_photon_t>(other) {}

Photons::Photons(Photons&& other)
    : ParticleBase<single_photon_t>(std::move(other)) {}

Photons::~Photons() {}

void
Photons::put(std::size_t pos, Pos_t x, Scalar p, Scalar path_left, int cell, int flag) {
  if (pos >= m_numMax)
    throw std::runtime_error("Trying to insert photon beyond the end of the array. Resize it first!");

  m_data.x1[pos] = x;
  m_data.p1[pos] = p;
  m_data.cell[pos] = cell;
  m_data.flag[pos] = flag;
  m_data.path_left[pos] = path_left;
  if (pos >= m_number) m_number = pos + 1;
}

void
Photons::append(Pos_t x, Scalar p, Scalar path_left, int cell, int flag) {
  put(m_number, x, p, path_left, cell, flag);
}

void
Photons::convert_pairs(Particles& electrons, Particles& positrons) {
  if (!create_pairs || !trace_photons)
    return;

  if (m_number <= 0)
    return;

  for (Index_t idx = 0; idx < m_number; idx++) {
    if (is_empty(idx))
      continue;

    if (m_data.path_left[idx] < 0.0) {
    // if (m_dist(m_generator) < p_ph) {
      double E_ph = std::abs(m_data.p1[idx]);
      double p_sec = sqrt(0.25 * E_ph * E_ph - 1.0);

      electrons.append(m_data.x1[idx], sgn(m_data.p1[idx]) * p_sec, m_data.cell[idx],
                       (check_flag(idx, PhotonFlag::tracked) ? (uint32_t)ParticleFlag::tracked : 0));
      positrons.append(m_data.x1[idx], sgn(m_data.p1[idx]) * p_sec, m_data.cell[idx],
                       (check_flag(idx, PhotonFlag::tracked) ? (uint32_t)ParticleFlag::tracked : 0));
      erase(idx);
    }
  }
}

void
Photons::sort(const Grid& grid) {
  if (m_number > 0)
    partition_and_sort(m_partition, grid, 8);
}

void
Photons::emit_photons(Particles &electrons, Particles &positrons) {
  if (!create_pairs)
    return;
  // This is assuming not in KN regime
  double E_ph = 20.0;
  // double l_photon = l_ph + 0.5 * l_ph * m_normal(m_generator);
  // double l_photon = 9999.9;
  double l_photon = -l_ph * std::log(1.0 - m_dist(m_generator));
  Logger::print_info("Processing Pair Creation...");
  // instant pair creation
  for (Index_t n = 0; n < electrons.number(); n++) {
    if (electrons.is_empty(n))
      continue;
    float gamma_ratio = electrons.data().gamma[n] / gamma_thr;
    if (gamma_ratio > 1.0) {
      float prob = (electrons.data().gamma[n] * e_min < 0.1 ? p_ic : p_ic * 0.1 / (e_min * electrons.data().gamma[n]));
      if (m_dist(m_generator) > prob)
        continue;
      // Assuming in KN regime, the photon takes 9/10 of the original energy
      // E_ph = 0.6 * m_dist(m_generator) * electrons.data().gamma[n] + 2.0;
      // E_ph = 0.6 * electrons.data().gamma[n] + 2.0;
      E_ph = draw_photon_energy(electrons.data().gamma[n], electrons.data().p1[n]);
      double gamma_f = electrons.data().gamma[n] - E_ph;
      if (gamma_f < 0.0)
        Logger::print_err("Photon energy exceeds particle energy! gamma is {}, Eph is {}", electrons.data().gamma[n], E_ph);
      if (gamma_f < 2.0) gamma_f = std::min(2.0, electrons.data().gamma[n]);
      double p_i = std::abs(electrons.data().p1[n]);
      electrons.data().p1[n] *= sqrt(gamma_f * gamma_f - 1.0) / p_i;
      if (E_ph * e_min < 0.01) continue;
      // track a fraction of the secondary particles and photons
      if (!trace_photons) {
        double p_sec = sqrt(0.25 * E_ph * E_ph - 1.0);
        electrons.append(electrons.data().x1[n], sgn(electrons.data().p1[n]) * p_sec,
                         electrons.data().cell[n],
                         (m_dist(m_generator) < track_pct ? (uint32_t)ParticleFlag::tracked : 0));
        positrons.append(electrons.data().x1[n], sgn(electrons.data().p1[n]) * p_sec,
                         electrons.data().cell[n],
                         (m_dist(m_generator) < track_pct ? (uint32_t)ParticleFlag::tracked : 0));
      } else {
        append(electrons.data().x1[n], sgn(electrons.data().p1[n]) * E_ph, l_photon,
               electrons.data().cell[n],
               // ((electrons.check_flag(n, ParticleFlag::tracked) && m_dist(m_generator) < track_pct) ?
                (m_dist(m_generator) < track_pct ? (uint32_t)PhotonFlag::tracked : 0));
      }
    }
  }
  for (Index_t n = 0; n < positrons.number(); n++) {
    if (positrons.is_empty(n))
      continue;
    float gamma_ratio = positrons.data().gamma[n] / gamma_thr;
    if (gamma_ratio > 1.0) {
      float prob = (positrons.data().gamma[n] * e_min < 0.1 ? p_ic : p_ic * 0.1 / (e_min * positrons.data().gamma[n]));
      if (m_dist(m_generator) > prob)
      // if (m_dist(m_generator) > p_ic * gamma_ratio)
        continue;
      // Assuming in KN regime, the photon takes 9/10 of the original energy
      // E_ph = 0.6 * m_dist(m_generator) * positrons.data().gamma[n] + 2.0;
      // E_ph = 0.6 * positrons.data().gamma[n] + 2.0;
      E_ph = draw_photon_energy(positrons.data().gamma[n], positrons.data().p1[n]);
      double gamma_f = positrons.data().gamma[n] - E_ph;
      if (gamma_f < 0.0)
        Logger::print_err("Photon energy exceeds particle energy! gamma is {}, Eph is {}", positrons.data().gamma[n], E_ph);
      if (gamma_f < 2.0) gamma_f = std::min(2.0, positrons.data().gamma[n]);
      double p_i = std::abs(positrons.data().p1[n]);
      positrons.data().p1[n] *= sqrt(gamma_f * gamma_f - 1.0) / p_i;
      if (E_ph * e_min < 0.01) continue;
      // track 10% of the secondary particles
      if (!trace_photons) {
        double p_sec = sqrt(0.25 * E_ph * E_ph - 1.0);
        electrons.append(positrons.data().x1[n], sgn(positrons.data().p1[n]) * p_sec,
                         positrons.data().cell[n],
                         ((m_dist(m_generator) < track_pct) ? (uint32_t)ParticleFlag::tracked : 0));
        positrons.append(positrons.data().x1[n], sgn(positrons.data().p1[n]) * p_sec,
                         positrons.data().cell[n],
                         (m_dist(m_generator) < track_pct ? (uint32_t)ParticleFlag::tracked : 0));
      } else {
        append(positrons.data().x1[n], sgn(positrons.data().p1[n]) * E_ph, l_photon,
               positrons.data().cell[n],
               // ((positrons.check_flag(n, ParticleFlag::tracked) && m_dist(m_generator) < track_pct) ?
               (m_dist(m_generator) < track_pct ? (uint32_t)PhotonFlag::tracked : 0));
      }
    }
  }
  Logger::print_info("There are now {} photons in the pool", m_number);
}

void
Photons::move(const Grid& grid, double dt) {
  auto& mesh = grid.mesh();

  for (Index_t idx = 0; idx < m_number; idx++) {
    if (is_empty(idx))
      continue;
    double p = m_data.p1[idx];
    int cell = m_data.cell[idx];
    double pos = mesh.pos(0, cell, m_data.x1[idx]);

    // Censor photons that are not converting inside the box
    // if (p < 0 && m_data.path_left[idx] > pos) {
    //   erase(idx);
    //   continue;
    // }
    // if (p > 0 && m_data.path_left[idx] > mesh.sizes[0] - pos) {
    //   erase(idx);
    //   continue;
    // }


    m_data.x1[idx] += sgn(p) * dt / mesh.delta[0];
    m_data.path_left[idx] -= dt;
    // Compute the change in particle cell
    auto c = mesh.get_cell_3d(cell);
    int delta_cell = (int)std::floor(m_data.x1[idx]);
    // std::cout << delta_cell << std::endl;
    c[0] += delta_cell;
    // Logger::print_info("After move, c is {}, x1 is {}", c, m_data.x1[idx]);

    m_data.cell[idx] = mesh.get_idx(c[0], c[1], c[2]);
    // std::cout << m_data.x1[idx] << ", " << m_data.cell[idx] << std::endl;
    m_data.x1[idx] -= (Pos_t)delta_cell;
    // std::cout << m_data.x1[idx] << ", " << m_data.cell[idx] << std::endl;
  }
}

void
Photons::compute_A1(double er) {
  // double er = 2.0 * gamma * e_min;
  A1 = 1.0 / (er * (0.5 + 1.0 / alpha - (1.0 / (alpha * (alpha + 1.0))) * std::pow(er / e_s, alpha)));
}

void
Photons::compute_A2(double er, double et) {
  // double er = 2.0 * gamma * e_min;
  // double et = er / (2.0 * er + 1.0);

  A2 = 1.0 / (et * (et * 0.5 / er + std::log(er / et) + 1.0 / (1.0 + alpha)));
}

double
Photons::f_inv1(double u, double gamma) {
  double er = 2.0 * gamma * e_min;
  compute_A1(er);
  if (u < A1 * er * 0.5)
    return std::sqrt(2.0 * u * er / A1);
  else if (u < 1.0 - A1 * er * std::pow(e_s / er, -alpha) / (1.0 + alpha))
    return er * std::pow(alpha * (1.0 / alpha + 0.5 - u / (A1 * er)), -1.0 / alpha);
  else
    return er * std::pow((1.0 - u)*(1.0 + alpha) / (A1 * e_s), -1.0 / (alpha + 1.0));
}

double
Photons::f_inv2(double u, double gamma) {
  double er = 2.0 * gamma * e_min;
  double et = er / (2.0 * er + 1.0);
  compute_A2(er, et);
  if (u < A2 * et * et * 0.5 / er)
    return std::sqrt(2.0 * u * er / A2);
  else if (u < A2 * et)
    return et * std::exp(u - A2 * et * et * 0.5 / er);
  else
    return er * std::pow((1.0 - u) / (A2 * et), -1.0 / (alpha + 1.0));
}

double
Photons::draw_photon_energy(double gamma, double p) {
  float u = m_dist(m_generator);
  // draw the rest frame photon energy
  double e1p;
  if (gamma < e_s * 0.5 / e_min) {
    e1p = f_inv1(u, gamma);
  } else {
    e1p = f_inv2(u, gamma);
  }
  // given energy, draw the rest frame photon angle
  u = m_dist(m_generator);
  double u1p;
  if (e1p < 0.5) {
    double q = std::pow(1.0 - 2.0 * e1p, 2.0 + alpha);
    u1p = (std::pow(u * (1.0 - q) + q, 1.0 / (2.0 + alpha)) + e1p - 1.0) / e1p;
  } else {
    u1p = 1.0 - (1.0 - std::pow(u, 1.0 / (2.0 + alpha))) / e1p;
  }
  // given e1p and u1p, compute the photon energy in the lab frame
  // Logger::print_info("e1p is {}, u1p is {}", e1p, u1p);
  return (gamma + std::abs(p) * (-u1p)) * e1p;
}

}
