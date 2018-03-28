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
  float p = std::sqrt(gamma * gamma - 1.0 - 0.8);
  for (int i = 0; i < N; i++) {
    std::cout << ph.draw_photon_energy(gamma, p) << std::endl;
  }

  return 0;
}
