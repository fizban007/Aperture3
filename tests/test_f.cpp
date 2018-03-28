#include "algorithms/functions.h"
#include "data/photons.h"
#include "sim_environment.h"
#include <iostream>

using namespace Aperture;

int main(int argc, char *argv[])
{
  Environment env(&argc, &argv);
  Photons ph(env);

  const int N = 100;
  float gamma = 300.0;
  float x = 0.9;
  float beta = beta_phi(x);
  float p = -std::sqrt(gamma * gamma - 1.0 - beta*beta);
  for (int i = 0; i < N; i++) {
    std::cout << ph.draw_photon_energy(gamma, p, x) << std::endl;
  }

  return 0;
}
