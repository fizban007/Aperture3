#include "ptc_updater_default.h"
#include "algorithms/interpolation.h"
#include "omp.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/simd.h"

namespace Aperture {

ptc_updater_default::ptc_updater_default(const sim_environment &env)
    : ptc_updater(env) {}

ptc_updater_default::~ptc_updater_default() {}

void
ptc_updater_default::update_particles(sim_data &data, double dt) {
  auto &ptc = data.particles;
  if (ptc.number() > 0) {
    // TODO: Boris push
#ifdef __AVX2__
    for (size_t idx = 0; idx < ptc.number(); idx += 8) {
      // Interpolate field values to particle position
      Vec8ui c;
      c.load_a(ptc.data().cell + idx);
      Vec8f x1;
      x1.load_a(ptc.data().x1 + idx);
      Vec8f x2;
      x2.load_a(ptc.data().x2 + idx);
      Vec8f x3;
      x3.load_a(ptc.data().x3 + idx);

      float q_over_m = dt * 0.5f;

      Vec8f E1 = interpolate(data.E.data(0), c, x1, x2, x3,
                             data.E.stagger(0)) *
                 q_over_m;
      Vec8f E2 = interpolate(data.E.data(1), c, x1, x2, x3,
                             data.E.stagger(1)) *
                 q_over_m;
      Vec8f E3 = interpolate(data.E.data(2), c, x1, x2, x3,
                             data.E.stagger(2)) *
                 q_over_m;
      Vec8f B1 = interpolate(data.B.data(0), c, x1, x2, x3,
                             data.B.stagger(0)) *
                 q_over_m;
      Vec8f B2 = interpolate(data.B.data(1), c, x1, x2, x3,
                             data.B.stagger(1)) *
                 q_over_m;
      Vec8f B3 = interpolate(data.B.data(2), c, x1, x2, x3,
                             data.B.stagger(2)) *
                 q_over_m;

      Vec8f p1;
      p1.load_a(ptc.data().p1 + idx);
      Vec8f p2;
      p2.load_a(ptc.data().p2 + idx);
      Vec8f p3;
      p3.load_a(ptc.data().p3 + idx);

      Vec8f pm1 = p1 + E1;
      Vec8f pm2 = p2 + E2;
      Vec8f pm3 = p3 + E3;
      Vec8f gamma = sqrt(pm1*pm1 + pm2*pm2 + pm3*pm3 + 1.0f);

      gamma.store_a(ptc.data().E + idx);

      Vec8f pp1 = pm1 + (pm2 * B3 - pm3 * B2) / gamma;
      Vec8f pp2 = pm2 + (pm3 * B1 - pm1 * B3) / gamma;
      Vec8f pp3 = pm3 + (pm1 * B2 - pm2 * B1) / gamma;
      Vec8f t2p1 = (B1*B1 + B2*B2 + B3*B3) / (gamma*gamma) + 1.0f;

      p1 = E1 + pm1 + (pp2 * B3 - pp3 * B2) / t2p1 * 2.0f;
      p2 = E2 + pm2 + (pp3 * B1 - pp1 * B3) / t2p1 * 2.0f;
      p3 = E3 + pm3 + (pp1 * B2 - pp2 * B1) / t2p1 * 2.0f;

      p1.store_a(ptc.data().p1 + idx);
      p2.store_a(ptc.data().p2 + idx);
      p3.store_a(ptc.data().p3 + idx);
#else
    for (size_t idx = 0; idx < ptc.number(); idx++) {
#endif
    }

    // TODO: Current deposit

    // TODO: Push photons as well
  }
}

#if defined(__AVX2__)
#endif

void
ptc_updater_default::handle_boundary(sim_data &data) {}

}  // namespace Aperture
