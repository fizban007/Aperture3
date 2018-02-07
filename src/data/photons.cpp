#include "data/photons.h"
#include "sim_environment.h"
#include "utils/Logger.h"
#include "utils/util_functions.h"

namespace Aperture {

Photons::Photons() {}

Photons::Photons(std::size_t max_num)
    : ParticleBase<single_photon_t>(max_num) {
  // No environment provided, all pair creation parameters going to be default
}

Photons::Photons(const Environment& env)
    : ParticleBase<single_photon_t>((std::size_t)env.conf().max_photon_number) {
  create_pairs = env.conf().create_pairs;
  trace_photons = env.conf().trace_photons;
  gamma_thr = env.conf().gamma_thr;
  l_ph = env.conf().photon_path;
}

Photons::Photons(const Photons& other)
    : ParticleBase<single_photon_t>(other) {}

Photons::Photons(Photons&& other)
    : ParticleBase<single_photon_t>(std::move(other)) {}

Photons::~Photons() {}

void
Photons::put(std::size_t pos, Pos_t x, Scalar p, int cell, int flag) {
  if (pos >= m_numMax)
    throw std::runtime_error("Trying to insert photon beyond the end of the array. Resize it first!");

  m_data.x1[pos] = x;
  m_data.p1[pos] = p;
  m_data.cell[pos] = cell;
  m_data.flag[pos] = flag;
  if (pos >= m_number) m_number = pos + 1;
}

void
Photons::append(Pos_t x, Scalar p, int cell, int flag) {
  put(m_number, x, p, cell, flag);
}

void
Photons::convert_pairs(Particles& electrons, Particles& positrons) {
  if (!create_pairs || !trace_photons)
    return;


}

void
Photons::make_pair(Index_t pos, Particles& electrons, Particles& positrons) {}

void
Photons::emit_photons(Particles &electrons, Particles &positrons) {
  if (!create_pairs)
    return;
  double E_ph = 3.0;
  Logger::print_info("Processing Pair Creation...");
  if (!trace_photons) {
    // instant pair creation
    for (Index_t n = 0; n < electrons.number(); n++) {
      if (electrons.data().gamma[n] > gamma_thr) {
        double gamma_f = electrons.data().gamma[n] - E_ph;
        double p_sec = sqrt(0.25 * E_ph * E_ph - 1.0);
        // electrons.append(electrons.data().x1[n], sgn(electrons.data().p1[n]) * p_sec,
        //                  electrons.data().cell[n],
        //                  (electrons.check_flag(n, ParticleFlag::tracked) ? ));
      }
    }
  } else {
    // create photons and then have them convert to pairs
  }
}

}
