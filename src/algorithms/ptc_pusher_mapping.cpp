#include "algorithms/ptc_pusher_mapping.h"
#include "metrics.h"

using namespace Aperture;

template <>
void
gravity_force::operator()<metric::metric_log_spherical>(
    const metric::metric_log_spherical &metric, Vec3<Mom_t> &result,
    const Vec3<Scalar> &pos, double dt) const {
  double r = std::exp(pos.x);
  result = Vec3<Scalar>(-m_g0 * dt / (r * r), 0.0, 0.0);
}

ParticlePusher_Mapping::ParticlePusher_Mapping(int interp_order)
    : m_order(interp_order), m_interp(interp_order) {
  m_algorithm = ForceAlgorithm::Vay;
  m_gravity = false;
  m_radiation = false;
}

ParticlePusher_Mapping::~ParticlePusher_Mapping() {}

void
ParticlePusher_Mapping::push(Aperture::SimData &data, double dt) {
  for (unsigned int i = 0; i < data.particles.size(); i++) {
    push(data.particles[i], data.E, data.B, dt);
  }
}

void
ParticlePusher_Mapping::push(Aperture::Particles &particles,
                             const vfield &E, const vfield &B,
                             double dt) {
  MetricType type = E.grid().type();
  // if (data.E.grid())
  // for (unsigned int i = 0; i < particles.size(); i++) {
  update_momentum(particles, E, B, dt);
#if defined(__AVX2__) && (defined(__ICC) || defined(__INTEL_COMPILER))
  select_metric(type, this->update_position_avx, *this, particles,
                E.grid(), dt);
#else
  select_metric(type, this->update_position, particles, E.grid(), dt, 0,
                particles.number());
#endif
  // }
}

void
ParticlePusher_Mapping::update_momentum(Aperture::Particles &particles,
                                        const vfield &E,
                                        const vfield &B, double dt) {
  update_momentum(particles, E, B, dt, 0, particles.number());
}

void
ParticlePusher_Mapping::update_momentum(Aperture::Particles &particles,
                                        const vfield &E,
                                        const vfield &B, double dt,
                                        Index_t start, Index_t num) {
  if (num < 1) return;

  auto &grid = E.grid();
  auto &mesh = grid.mesh();
  for (Index_t idx = start;
       idx < std::min(start + num, particles.number()); idx++) {
    if (particles.is_empty(idx)) continue;
    auto &ptc = particles.data();
    if (check_bit(ptc.flag[idx], ParticleFlag::ignore_force)) continue;
    Vec3<Mom_t> dp;
    Vec3<float> x(ptc.x1[idx], ptc.x2[idx], ptc.x3[idx]);
    Vec3<double> p(ptc.p1[idx], ptc.p2[idx], ptc.p3[idx]);
    auto pos = mesh.pos_particle(ptc.cell[idx], x);

    // Gravity force
    // if (m_gravity) {
    //   select_metric(m_g, grid.type(), dp, pos, dt);
    // }

    // Lorentz force
    if (!check_bit(ptc.flag[idx], ParticleFlag::ignore_EM)) {
      // Vec3<Scalar> vE = m_interp.interp_cell(ptc.x[idx].vec3(),
      // grid.);
      auto c = mesh.get_cell_3d(ptc.cell[idx]);
      Vec3<Scalar> vE = E.interpolate(c, x, m_order);
      Vec3<Scalar> vB = B.interpolate(c, x, m_order);
      if (m_algorithm == ForceAlgorithm::Vay)
        dp += vay_push(p, vE, vB, particles.charge() / particles.mass(),
                       dt);
      else if (m_algorithm == ForceAlgorithm::Boris)
        dp += boris_push(p, vE, vB,
                         particles.charge() / particles.mass(), dt);
    }

    // TODO: Radiation force
    if (m_radiation) {
    }

    // Update momentum
    ptc.p1[idx] += dp.x;
    ptc.p2[idx] += dp.y;
    ptc.p3[idx] += dp.z;
    double p_mag =
        sqrt(ptc.p1[idx] * ptc.p1[idx] + ptc.p2[idx] * ptc.p2[idx] +
             ptc.p3[idx] * ptc.p3[idx]);
    ptc.gamma[idx] = sqrt(1.0 + p_mag * p_mag);
    p.x = ptc.p1[idx];
    p.y = ptc.p2[idx];
    p.z = ptc.p3[idx];

    if (m_compute_curvature) {
      // Compute radius of curvature
      // TODO: Check the correctness of this line!
      auto p_cross_dp = p.cross(dp);
      ptc.Rc[idx] = dt * p_mag * p_mag * p_mag /
                    (ptc.gamma[idx] * p_cross_dp.length());
    }
  }
}

void
ParticlePusher_Mapping::update_momentum_avx(
    Aperture::Particles &particles, const vfield &E, const vfield &B,
    double dt) {
#if defined(__AVX2__) && (defined(__ICC) || defined(__INTEL_COMPILER))
  auto &grid = E.grid();
  auto &mesh = grid.mesh();

  Index_t idx = 0;
  auto &ptc = particles.data();
  for (idx = 0; idx < particles.number(); idx += 4) {
    // __m256d x1 = _mm256_load_pd(ptc.x1 + idx);
    // __m256d x2 = _mm256_load_pd(ptc.x2 + idx);
    // __m256d x3 = _mm256_load_pd(ptc.x3 + idx);
    __m256d p1 = _mm256_load_pd(ptc.p1 + idx);
    __m256d p2 = _mm256_load_pd(ptc.p2 + idx);
    __m256d p3 = _mm256_load_pd(ptc.p3 + idx);
    // __m128i cell = _mm128_load_si128(ptc.cell + idx);

    // Interpolate the field to generate 6 field registers
  }

  update_momentum(particles, E, B, dt, idx, particles.number() - idx);
#endif
}

#if defined(__AVX2__) && (defined(__ICC) || defined(__INTEL_COMPILER))
void
ParticlePusher_Mapping::vay_push_avx(__m256d *p1, __m256d *p2,
                                     __m256d *p3, __m256d *gamma,
                                     __m256d E1, __m256d E2, __m256d E3,
                                     __m256d B1, __m256d B2, __m256d B3,
                                     double q_over_m, double dt) {
  // q_over_m *= dt * 0.5;
  // Vec3<Scalar> u_halfstep = p + (E + p.cross(B) / p.length() ) *
  // q_over_m;
  __m256d q_over_m_pd = _mm256_set1_pd(q_over_m * dt * 0.5);
  B1 = _mm256_mul_pd(B1, q_over_m_pd);
  B2 = _mm256_mul_pd(B2, q_over_m_pd);
  B3 = _mm256_mul_pd(B3, q_over_m_pd);

  __m256d cross1 = _mm256_div_pd(
      _mm256_sub_pd(_mm256_mul_pd(*p2, B3), _mm256_mul_pd(*p3, B2)),
      *gamma);
  __m256d cross2 = _mm256_div_pd(
      _mm256_sub_pd(_mm256_mul_pd(*p3, B1), _mm256_mul_pd(*p1, B3)),
      *gamma);
  __m256d cross3 = _mm256_div_pd(
      _mm256_sub_pd(_mm256_mul_pd(*p1, B2), _mm256_mul_pd(*p2, B1)),
      *gamma);
  __m256d u_prime1 = _mm256_fmadd_pd(E1, _mm256_set1_pd(q_over_m * dt),
                                     _mm256_add_pd(*p1, cross1));
  __m256d u_prime2 = _mm256_fmadd_pd(E2, _mm256_set1_pd(q_over_m * dt),
                                     _mm256_add_pd(*p2, cross2));
  __m256d u_prime3 = _mm256_fmadd_pd(E3, _mm256_set1_pd(q_over_m * dt),
                                     _mm256_add_pd(*p3, cross3));

  // Vec3<Scalar> upr = u_halfstep + E * q_over_m;
  // Vec3<Scalar> tau = B * q_over_m;

  // store some repeatedly used intermediate results
  __m256d tt = _mm256_fmadd_pd(
      B1, B1, _mm256_fmadd_pd(B2, B2, _mm256_mul_pd(B3, B3)));
  // Scalar ut = upr.dot( tau );
  __m256d ut = _mm256_fmadd_pd(
      u_prime1, B1,
      _mm256_fmadd_pd(u_prime2, B2, _mm256_mul_pd(u_prime3, B3)));

  // Scalar sigma = 1.0 + upr.dot(upr) - tt;
  __m256d sigma = _mm256_sub_pd(
      _mm256_add_pd(
          _mm256_set1_pd(1.0),
          _mm256_fmadd_pd(
              u_prime1, u_prime1,
              _mm256_fmadd_pd(u_prime2, u_prime2,
                              _mm256_mul_pd(u_prime3, u_prime3)))),
      tt);
  // inv_gamma2 means ( 1 / gamma^(i+1) ) ^2
  // Scalar inv_gamma2 =  2.0 / ( sigma + std::sqrt( sigma * sigma + 4.0
  // * ( tt + ut * ut ) ) );
  *gamma = _mm256_sqrt_pd(_mm256_mul_pd(
      _mm256_set1_pd(0.5),
      _mm256_add_pd(sigma,
                    _mm256_sqrt_pd(_mm256_fmadd_pd(
                        sigma, sigma,
                        _mm256_mul_pd(_mm256_set1_pd(4.0),
                                      _mm256_fmadd_pd(ut, ut, tt)))))));

  // Scalar s = 1.0 / ( 1.0 + inv_gamma2 * tt );
  __m256d inv_gamma = _mm256_div_pd(_mm256_set1_pd(1.0), *gamma);
  __m256d s = _mm256_fmadd_pd(tt, _mm256_mul_pd(inv_gamma, inv_gamma),
                              _mm256_set1_pd(1.0));
  // result = upr.cross(tau);
  *p1 = _mm256_sub_pd(_mm256_mul_pd(u_prime2, B3),
                      _mm256_mul_pd(u_prime3, B2));
  *p2 = _mm256_sub_pd(_mm256_mul_pd(u_prime3, B1),
                      _mm256_mul_pd(u_prime1, B3));
  *p3 = _mm256_sub_pd(_mm256_mul_pd(u_prime1, B2),
                      _mm256_mul_pd(u_prime2, B1));
  *p1 = _mm256_fmadd_pd(
      *p1, inv_gamma,
      _mm256_fmadd_pd(
          B1, _mm256_mul_pd(ut, _mm256_mul_pd(inv_gamma, inv_gamma)),
          u_prime1));
  *p2 = _mm256_fmadd_pd(
      *p2, inv_gamma,
      _mm256_fmadd_pd(
          B2, _mm256_mul_pd(ut, _mm256_mul_pd(inv_gamma, inv_gamma)),
          u_prime2));
  *p3 = _mm256_fmadd_pd(
      *p3, inv_gamma,
      _mm256_fmadd_pd(
          B3, _mm256_mul_pd(ut, _mm256_mul_pd(inv_gamma, inv_gamma)),
          u_prime3));
  // result *= std::sqrt(inv_gamma2);
  // result += upr + tau * (ut * inv_gamma2);
  // p_vay += tau * (ut * inv_gamma2);
  *p1 = _mm256_div_pd(*p1, s);
  *p2 = _mm256_div_pd(*p2, s);
  *p3 = _mm256_div_pd(*p3, s);
  // result -= p;
  // Vec3<Mom_t> p_vay = ( upr + tau * ( ut * inv_gamma2 ) + upr.cross(
  // tau ) * std::sqrt(inv_gamma2) ) * s; return result;
}
#endif
// void
// ParticlePusher_Mapping::up

// ParticlePusher_Mapping&
// ParticlePusher_Mapping::set_algorithm(Aperture::ForceAlgorithm
// algorithm) {
//   m_algorithm = algorithm;
//   return *this;
// }

// ParticlePusher_Mapping&
// ParticlePusher_Mapping::set_gravity(double g) {
//   m_gravity = true;
//   m_g.m_g0 = g;
//   return *this;
// }

// ParticlePusher_Mapping&
// ParticlePusher_Mapping::set_radiation(bool radiation) {
//   m_radiation = radiation;
//   return *this;
// }

// ParticlePusher_Mapping&
// ParticlePusher_Mapping::set_compute_curvature(bool c) {
//   m_compute_curvature = c;
//   return *this;
// }

Vec3<Mom_t>
ParticlePusher_Mapping::vay_push(const Vec3<Mom_t> &p,
                                 const Vec3<Scalar> &E,
                                 const Vec3<Scalar> &B, double q_over_m,
                                 double dt) {
  Scalar lambda =
      q_over_m * dt / 2.0;  // measured in units of (e/m) * R_* / c;
  Vec3<Scalar> u_halfstep =
      p + E * lambda +
      p.cross(B) * (lambda / std::sqrt(1.0 + p.dot(p)));
  Vec3<Scalar> upr = u_halfstep + E * lambda;
  Vec3<Scalar> tau = B * lambda;
  // store some repeatedly used intermediate results
  Scalar tt = tau.dot(tau);
  Scalar ut = upr.dot(tau);

  Scalar sigma = 1.0 + upr.dot(upr) - tt;
  // inv_gamma2 means ( 1 / gamma^(i+1) ) ^2
  Scalar inv_gamma2 =
      2.0 / (sigma + std::sqrt(sigma * sigma + 4.0 * (tt + ut * ut)));
  Scalar s = 1.0 / (1.0 + inv_gamma2 * tt);
  Vec3<Mom_t> p_vay = (upr + tau * (ut * inv_gamma2) +
                       upr.cross(tau) * std::sqrt(inv_gamma2)) *
                      s;
  //    Vec3<MOM_TYPE> dp = p_vay - p;
  return p_vay - p;
  // Vec3<Mom_t> result;
  // q_over_m *= dt * 0.5;
  // Vec3<Scalar> u_halfstep = p + (E + p.cross(B) / p.length() ) *
  // q_over_m; Vec3<Scalar> upr = u_halfstep + E * q_over_m;
  // Vec3<Scalar> tau = B * q_over_m;
  // // store some repeatedly used intermediate results
  // Scalar tt = tau.dot( tau );
  // Scalar ut = upr.dot( tau );

  // Scalar sigma = 1.0 + upr.dot(upr) - tt;
  // // inv_gamma2 means ( 1 / gamma^(i+1) ) ^2
  // Scalar inv_gamma2 =  2.0 / ( sigma + std::sqrt( sigma * sigma + 4.0
  // * ( tt + ut * ut ) ) ); Scalar s = 1.0 / ( 1.0 + inv_gamma2 * tt );
  // result = upr.cross(tau);
  // result *= std::sqrt(inv_gamma2);
  // result += upr + tau * (ut * inv_gamma2);
  // // p_vay += tau * (ut * inv_gamma2);
  // result *= s;
  // result -= p;
  // // Vec3<Mom_t> p_vay = ( upr + tau * ( ut * inv_gamma2 ) +
  // upr.cross( tau ) * std::sqrt(inv_gamma2) ) * s; return result;
}

Vec3<Mom_t>
ParticlePusher_Mapping::boris_push(const Vec3<Mom_t> &p,
                                   const Vec3<Scalar> &E,
                                   const Vec3<Scalar> &B,
                                   double q_over_m, double dt) {
  Vec3<Mom_t> result;
  return result;
}
