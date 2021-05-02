#ifndef _SPECTRA_H_
#define _SPECTRA_H_

#include "core/typedefs.h"

namespace Aperture {

namespace Spectra {

struct power_law_hard {
  HOST_DEVICE power_law_hard(Scalar delta, Scalar emin, Scalar emax)
      : delta_(delta), emin_(emin), emax_(emax) {}

  HD_INLINE Scalar operator()(Scalar e) const {
    if (e < emax_ && e > emin_)
      return pow(e / emax_, delta_) / e;
    else
      return 0.0;
  }

  Scalar emin() const { return emin_; }
  Scalar emax() const { return emax_; }

  Scalar delta_, emin_, emax_;
};

struct power_law_soft {
  HOST_DEVICE power_law_soft(Scalar alpha, Scalar emin, Scalar emax)
      : alpha_(alpha), emin_(emin), emax_(emax) {}

  HD_INLINE Scalar operator()(Scalar e) const {
    if (e < emax_ && e > emin_)
      return pow(e / emin_, -alpha_) / e;
    else
      return 0.0;
  }

  Scalar emin() const { return emin_; }
  Scalar emax() const { return emax_; }

  Scalar alpha_, emin_, emax_;
};

struct broken_power_law {
  HOST_DEVICE broken_power_law(Scalar alpha, Scalar delta, Scalar ep,
                               Scalar emin, Scalar emax)
      : alpha_(alpha),
        delta_(delta),
        epeak_(ep),
        emin_(emin),
        emax_(emax) {}

  HD_INLINE Scalar operator()(Scalar e) const {
    if (e < epeak_ && e > emin_)
      return pow(e / epeak_, delta_) / e;
    else if (e > epeak_ && e < emax_)
      return pow(e / epeak_, -alpha_) / e;
    else
      return 0.0;
  }

  Scalar emin() const { return emin_; }
  Scalar emax() const { return emax_; }

  Scalar alpha_, delta_, epeak_, emin_, emax_;
};

struct black_body {
  HOST_DEVICE black_body(Scalar kT) : kT_(kT) {}

  HD_INLINE double operator()(double e) const {
    // The normalization factor comes as 8 \pi/(h^3 c^3) (me c^2)^3
    // return 1.75464e30 * e * e / (exp(e / kT_) - 1.0);
    return e * e / (exp(e / kT_) - 1.0) / (2.4 * kT_ * kT_ * kT_);
  }

  Scalar emin() const { return 1e-10 * kT_; }
  Scalar emax() const { return 1e3 * kT_; }

  Scalar kT_;
};

struct mono_energetic {
  HOST_DEVICE mono_energetic(Scalar e0, Scalar de) : e0_(e0), de_(de) {}

  HD_INLINE Scalar operator()(Scalar e) const {
    if (e < e0_ + de_ && e > e0_ - de_)
      return 1.0 / (2.0 * de_) / e;
    else
      return 0.0;
  }

  Scalar emin() const { return e0_ * 1e-4; }
  Scalar emax() const { return e0_ * 1e4; }

  Scalar e0_, de_;
};

}  // namespace Spectra

}  // namespace Aperture

#endif  // _SPECTRA_H_
