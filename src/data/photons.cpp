#include "data/photons.h"
#include "sim_environment.h"
#include "utils/Logger.h"
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
  track_pct = env.conf().track_percent;
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
Photons::make_pair(Index_t pos, Particles& electrons, Particles& positrons) {
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
  double E_ph = 3.0;
  Logger::print_info("Processing Pair Creation...");
  // instant pair creation
  for (Index_t n = 0; n < electrons.number(); n++) {
    if (electrons.is_empty(n))
      continue;
    if (electrons.data().gamma[n] > gamma_thr) {
      double gamma_f = electrons.data().gamma[n] - E_ph;
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
        append(electrons.data().x1[n], sgn(electrons.data().p1[n]) * E_ph, l_ph,
               electrons.data().cell[n],
               // ((electrons.check_flag(n, ParticleFlag::tracked) && m_dist(m_generator) < track_pct) ?
                (m_dist(m_generator) < track_pct ? (uint32_t)PhotonFlag::tracked : 0));
      }
      double p_i = std::abs(electrons.data().p1[n]);
      electrons.data().p1[n] *= sqrt(gamma_f * gamma_f - 1.0) / p_i;
    }
  }
  for (Index_t n = 0; n < positrons.number(); n++) {
    if (positrons.is_empty(n))
      continue;
    if (positrons.data().gamma[n] > gamma_thr) {
      double gamma_f = positrons.data().gamma[n] - E_ph;
      double p_sec = sqrt(0.25 * E_ph * E_ph - 1.0);
      // track 10% of the secondary particles
      if (!trace_photons) {
        electrons.append(positrons.data().x1[n], sgn(positrons.data().p1[n]) * p_sec,
                         positrons.data().cell[n],
                         ((m_dist(m_generator) < track_pct) ? (uint32_t)ParticleFlag::tracked : 0));
        positrons.append(positrons.data().x1[n], sgn(positrons.data().p1[n]) * p_sec,
                         positrons.data().cell[n],
                         (m_dist(m_generator) < track_pct ? (uint32_t)ParticleFlag::tracked : 0));
      } else {
        append(positrons.data().x1[n], sgn(positrons.data().p1[n]) * E_ph, l_ph,
               positrons.data().cell[n],
               // ((positrons.check_flag(n, ParticleFlag::tracked) && m_dist(m_generator) < track_pct) ?
               (m_dist(m_generator) < track_pct ? (uint32_t)PhotonFlag::tracked : 0));
      }
      double p_i = std::abs(positrons.data().p1[n]);
      positrons.data().p1[n] *= sqrt(gamma_f * gamma_f - 1.0) / p_i;
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
    int cell = m_data.cell[idx];

    m_data.x1[idx] += sgn(m_data.p1[idx]) * dt / mesh.delta[0];
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

}
