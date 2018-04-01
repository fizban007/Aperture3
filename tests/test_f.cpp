#include "algorithms/functions.h"
#include "data/photons.h"
#include "sim_environment.h"
#include <iostream>
#include <fstream>
#include <random>
#include <vector>

using namespace Aperture;

int main(int argc, char *argv[])
{
  Environment env(&argc, &argv);
  Photons ph(env);

  std::cout << "e_min is " << env.conf().e_min << std::endl;
  const int N = 100000;
  float gamma = 1.0e3;
  std::cout << "e_min * gamma is " << env.conf().e_min * gamma << std::endl;
  float x = 0.5;
  float beta = beta_phi(x);
  float p = std::sqrt(gamma * gamma - 1.0 - beta*beta);

  float emin = 1.0e-5;
  float alpha = 2.0;

  std::default_random_engine g;
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  std::ofstream f2("angles.txt");
  std::ofstream f3("e1p.txt");
  // f << "e_min * gamma is " << env.conf().e_min * gamma << std::endl;
  // std::vector<float> v_g {1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7};
  // for (auto gamma : v_g) {
    std::ofstream f("spectrum1e3.txt");
    for (int i = 0; i < N; i++) {
      // float E_ph = ph.draw_photon_e1p(gamma);
      // float e1p = ph.draw_photon_e1p(gamma);
      float e1p = 25.0*gamma*emin;
      float u1p = ph.draw_photon_u1p(e1p, gamma);
      float ep = ph.draw_photon_ep(e1p, gamma);
      float E_ph = ph.draw_photon_energy(gamma, p, x) / gamma;
      // std::cout << ph.draw_photon_energy(gamma, p, x) << std::endl;

      // float u = dist(g);
      // float E_target = emin * std::pow(1.0 - u, -1.0 / alpha);
      // float E_prime = E_target * gamma / 2.0;
      f << E_ph << std::endl;
      // double f_inv2 = ph.f_inv2((double)i / (double)N, gamma);
      // f << f_inv2 << std::endl;
      // float u1 = ph.draw_photon_u1p(200.0);
      f2 << u1p << std::endl;
      f3 << ep << std::endl;
    }
  // }
  f.close();
  f2.close();
  f3.close();

  return 0;
}
