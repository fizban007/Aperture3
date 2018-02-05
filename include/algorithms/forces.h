#ifndef _FORCES_H_
#define _FORCES_H_

#include "data/typedefs.h"
#include "data/vec3.h"
#include "metrics.h"

namespace Aperture {

// Lorentz Force, implemented using Vay algorithm
struct Lorentz_force_Vay {
  Vec3<Mom_t> operator() (const Vec3<Mom_t>& p, const Vec3<Scalar>& E, const Vec3<Scalar>& B,
                          double q_over_m, double dt) {
    q_over_m *= dt * 0.5;
    Vec3<Scalar> u_halfstep = p + (E + p.cross(B) / p.length() ) * q_over_m;
    Vec3<Scalar> upr = u_halfstep + E * q_over_m;
    Vec3<Scalar> tau = B * q_over_m;
    // store some repeatedly used intermediate results
    Scalar tt = tau.dot( tau );
    Scalar ut = upr.dot( tau );

    Scalar sigma = 1.0 + upr.dot(upr) - tt;
    // inv_gamma2 means ( 1 / gamma^(i+1) ) ^2
    Scalar inv_gamma2 =  2.0 / ( sigma + std::sqrt( sigma * sigma + 4.0 * ( tt + ut * ut ) ) );
    Scalar s = 1.0 / ( 1.0 + inv_gamma2 * tt );
    Vec3<Mom_t> p_vay = upr.cross(tau);
    p_vay *= std::sqrt(inv_gamma2);
    p_vay += upr;
    p_vay += tau * (ut * inv_gamma2);
    p_vay *= s;
    // Vec3<Mom_t> p_vay = ( upr + tau * ( ut * inv_gamma2 ) + upr.cross( tau ) * std::sqrt(inv_gamma2) ) * s;
    return p_vay - p;
  }
};

// Lorentz Force, implemented using Boris algorithm
struct Lorentz_force_Boris {
  Vec3<Mom_t> operator() (const Vec3<Mom_t>& p, const Vec3<Scalar>& E, const Vec3<Scalar>& B,
                          double q_over_m, double dt) {
    // TODO: Finish the implementation
    Vec3<Mom_t> result;
    return result;
  }
};

struct gravity_force {
  gravity_force() : m_g0(0.0) {}
  gravity_force(double g0) : m_g0(g0) {}

  template <typename Metric>
  void operator() (const Metric& metric, Vec3<Mom_t>& result, const Vec3<Scalar>& pos, double dt) const;

  double m_g0;
};

}

#endif  // _FORCES_H_
