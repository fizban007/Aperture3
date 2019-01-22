#include "algorithms/functions.h"
#include "data/photons_dev.h"
#include "sim_environment_dev.h"
#include <iostream>
#include <fstream>
#include <random>
#include <vector>

using namespace Aperture;

int main(int argc, char *argv[])
{
  cu_sim_environment env(&argc, &argv);
  Photons ph(env);

  const int N = 1000;
  double Eph = 0.002 / env.conf().e_min;
  std::ofstream f("photon_paths.txt");
  for (int i = 0; i < N; i++) {
    double l = ph.draw_photon_freepath(Eph);
    f << l << std::endl;
  }
  f.close();
  return 0;
}
