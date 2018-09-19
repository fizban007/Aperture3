#include "algorithms/functions.h"

namespace Aperture {

double
beta_phi(double x) {
  double b = (x - 0.05) / 0.45 - 1.0;
  return b;
  // if (x > 0.05 && x < 0.95) {
  //   return b;
  // } else {
  //   return b;
  // }
}
}  // namespace Aperture
