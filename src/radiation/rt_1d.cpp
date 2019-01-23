#include "radiation/rt_1d.h"
#include "sim_data.h"
#include "sim_environment.h"
#include <cstdint>

namespace Aperture {

rad_transfer_1d::rad_transfer_1d(const sim_environment& env)
    : m_env(env), m_dist(0.0, 1.0) {}

rad_transfer_1d::~rad_transfer_1d() {}

void
rad_transfer_1d::emit_photons(sim_data& data) {
  auto& photons = data.photons;
  auto& ptc = data.particles;
  auto& ptcdata = ptc.data();
  auto& mesh = m_env.grid().mesh();

  uint32_t num_ph = 0;
  for (Index_t idx = 0; idx < ptc.number(); idx++) {
    Scalar gamma = ptcdata.E[idx];
    Scalar x = mesh.pos(0, ptcdata.cell[idx], ptcdata.x1[idx]);
    if (gamma > m_env.params().gamma_thr && x < 0.3 * mesh.sizes[0]) {
      Scalar p_i = std::abs(ptcdata.p1[idx]);
      Scalar E_f = gamma - 2.0 * m_env.params().E_secondary;
      Scalar l_ph = 0.0;
      photons.append(
          {ptcdata.x1[idx], 0.0, 0.0},
          {sgn(ptcdata.p1[idx]) * (Scalar)2.0 * m_env.params().E_secondary, 0.0,
           0.0},
          ptcdata.cell[idx], l_ph, ptcdata.weight[idx],
          (m_dist(m_gen) < 0.1 ? bit_or(PhotonFlag::tracked) : 0));
      ptcdata.p1[idx] *= sqrt(E_f * E_f - 1.0) / p_i;
      num_ph += 1;
    }
  }
  Logger::print_info("{} photons produced", num_ph);
}

void
rad_transfer_1d::produce_pairs(sim_data& data) {
  auto& photons = data.photons;
  auto& ptc = data.particles;
  auto& phdata = photons.data();

  for (Index_t idx = 0; idx < photons.number(); idx++) {
    if (phdata.cell[idx] != MAX_CELL && phdata.path_left[idx] <= 0.0) {
      Scalar p = phdata.p1[idx] / 2;
      p = sgn(p) * sqrt(p * p - 1.0);
      ptc.append(
          {phdata.x1[idx], 0.0, 0.0}, {p, 0.0, 0.0}, phdata.cell[idx],
          ParticleType::electron, phdata.weight[idx],
          (m_dist(m_gen) < 0.1 ? bit_or(ParticleFlag::tracked) : 0));
      ptc.append(
          {phdata.x1[idx], 0.0, 0.0}, {p, 0.0, 0.0}, phdata.cell[idx],
          ParticleType::positron, phdata.weight[idx],
          (m_dist(m_gen) < 0.1 ? bit_or(ParticleFlag::tracked) : 0));

      phdata.cell[idx] = MAX_CELL;
    }
  }
}

}  // namespace Aperture
