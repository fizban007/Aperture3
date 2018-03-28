#include "algorithms/functions.h"

namespace Aperture {

double beta_phi(double x) {
  double b = (x - 0.1)/0.4 - 1.0;
  if (x > 0.1 && x < 0.9) {
    return b;
  } else {
    return b;
  }
}
}
