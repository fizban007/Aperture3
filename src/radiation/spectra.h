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

struct black_body {
  HOST_DEVICE black_body(Scalar kT) : kT_(kT) {}

  HD_INLINE Scalar operator()(Scalar e) const {
    return e * e / (exp(e / kT_) - 1.0);
  }

  Scalar emin() const { return 1e-6*kT_; }
  Scalar emax() const { return 1e4*kT_; }

  Scalar kT_;
};

struct mono_energetic {
  HOST_DEVICE mono_energetic(Scalar e0, Scalar de)
      : e0_(e0), de_(de) {}

  HD_INLINE Scalar operator()(Scalar e) const {
    if (e < e0_ + de_ && e > e0_ - de_)
      return 1.0 / (2.0 * de_);
    else
      return 0.0;
  }

  Scalar emin() const { return e0_ * 1e-2; }
  Scalar emax() const { return e0_ * 1e2; }

  Scalar e0_, de_;
};

}  // namespace Spectra

}  // namespace Aperture

#endif  // _SPECTRA_H_
