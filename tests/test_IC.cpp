#include <cuda_runtime.h>
#include "cuda/radiation/rt_ic.h"
#include "radiation/spectra.h"
#include "sim_params.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/util_functions.h"
#include <boost/math/quadrature/gauss.hpp>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <boost/multi_array.hpp>
#define H5_USE_BOOST
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

using namespace Aperture;
using namespace HighFive;

Scalar e_min = 1.0e-3;
Scalar e_max = 1.0e6;

struct maxwellian {
  Scalar kT_;

  maxwellian(Scalar kT) : kT_(kT) {}

  Scalar operator()(Scalar gamma) const {
    Scalar beta = std::sqrt(1.0 - 1.0 / square(gamma));
    return gamma * gamma * beta * std::exp(-gamma / kT_);
  }

  Scalar emin() const { return 1.0; }
};

struct power_law {
  Scalar n_, e_;

  power_law(Scalar n, Scalar e) : n_(n), e_(e) {}

  Scalar operator()(Scalar gamma) const {
    return std::pow(gamma / e_, -n_);
  }

  Scalar emin() const { return e_; }
};

struct sample_gamma_dist {
  std::vector<Scalar> dist_;
  std::vector<Scalar> gammas_;
  Scalar dg_;
  Scalar g_min = 1.0e3;
  Scalar g_max = 1.0e6;

  template <typename F>
  sample_gamma_dist(const F& f) : dist_(500), gammas_(500) {
    g_min = f.emin();
    dg_ = (std::log(g_max) - std::log(g_min)) / (dist_.size() - 1.0);
    for (int n = 0; n < dist_.size(); n++) {
      gammas_[n] = std::exp(std::log(g_min) + n * dg_);
      if (n > 0)
        dist_[n] = dist_[n - 1] + f(gammas_[n]) * gammas_[n] * dg_;
      // Logger::print_info("dist is {}, f is {}, gamma is {}, kT is
      // {}", dist_[n], f(gammas_[n]), gammas_[n], f.kT_); Scalar beta =
      // std::sqrt(1.0 - 1.0 / square(gammas_[n]));
    }
    for (int n = 0; n < dist_.size(); n++) {
      dist_[n] /= dist_[dist_.size() - 1];
      // Logger::print_info("dist is {}", dist_[n]);
    }
  }

  Scalar inverse_dist(float u) {
    int a = 0, b = dist_.size() - 1, tmp = 0;
    Scalar v = 0.0f, l = 0.0f, h = 0.0f;
    while (a < b) {
      tmp = (a + b) / 2;
      v = dist_[tmp];
      if (v < u) {
        a = tmp + 1;
      } else if (v > u) {
        b = tmp - 1;
      } else {
        b = tmp;
        l = h = v;
        break;
      }
    }
    if (v < u) {
      l = v;
      h = (a < dist_.size() ? dist_[a] : v);
      b = tmp;
    } else {
      h = v;
      l = (b >= 0 ? dist_[b] : v);
      b = std::max(b, 0);
    }
    Scalar bb = (l == h ? b : (u - l) / (h - l) + b);
    return std::exp(std::log(g_min) + bb * dg_);
  }
};

int
main(int argc, char* argv[]) {
  SimParams params;
  params.n_gamma = 600;
  params.n_ep = 600;
  inverse_compton ic(params);

  std::default_random_engine gen;
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  // Spectra::power_law_hard ne(0.2, e_min, e_max);
  // Spectra::power_law_soft ne(2.0, e_min, e_max);
  Spectra::black_body ne(1.0e-3);
  // Spectra::broken_power_law ne(1.25, 2.0, 1.0e-3, 1.0e-7, 1.0e4);
  // Spectra::mono_energetic ne(0.001, 1.0e-4);
  ic.init(ne, ne.emin(), ne.emax());

  File datafile("spectrum.h5",
                File::ReadWrite | File::Create | File::Truncate);
  Logger::print_info("gammas size is {}", ic.gammas().size());
  DataSet data_gammas = datafile.createDataSet<Scalar>(
      "gammas", DataSpace(ic.gammas().size()));
  data_gammas.write(ic.gammas().data());
  DataSet data_rates = datafile.createDataSet<Scalar>(
      "rates", DataSpace(ic.ic_rate().size()));
  data_rates.write(ic.ic_rate().data());
  DataSet data_gg_rates = datafile.createDataSet<Scalar>(
      "gg_rates", DataSpace(ic.gg_rate().size()));
  data_gg_rates.write(ic.gg_rate().data());
  DataSet data_ep =
      datafile.createDataSet<Scalar>("ep", DataSpace(ic.ep().size()));
  data_ep.write(ic.ep().data());

  const uint32_t N_samples = 10000000;
  cu_array<Scalar> g_M(N_samples);
  cu_array<Scalar> g_PL1(N_samples);
  cu_array<Scalar> g_PL2(N_samples);
  cu_array<Scalar> eph_M(N_samples);
  cu_array<Scalar> eph_PL1(N_samples);
  cu_array<Scalar> eph_PL2(N_samples);

  sample_gamma_dist DM(maxwellian(10.0));
  sample_gamma_dist DP1(power_law(1.0, 10.0));
  sample_gamma_dist DP2(power_law(2.0, 10.0));
  for (uint32_t i = 0; i < N_samples; i++) {
    float u = dist(gen);
    g_PL2[i] = DP2.inverse_dist(u);
    g_PL1[i] = DP1.inverse_dist(u);
    g_M[i] = DM.inverse_dist(u);
    // Logger::print_info("gamma is {}, u is {}", gammas[i], u);
  }
  DataSet data_energies = datafile.createDataSet<Scalar>(
      "e_Maxwellian", DataSpace(g_M.size()));
  data_energies.write(g_M.data());
  g_M.copy_to_device();
  DataSet data_PL1 =
      datafile.createDataSet<Scalar>("e_PL1", DataSpace(g_PL1.size()));
  data_PL1.write(g_PL1.data());
  g_PL1.copy_to_device();
  DataSet data_PL2 =
      datafile.createDataSet<Scalar>("e_PL2", DataSpace(g_PL2.size()));
  data_PL2.write(g_PL2.data());
  g_PL2.copy_to_device();
  cudaDeviceSynchronize();
  // timer::stamp();
  ic.generate_photon_energies(eph_M, g_M);
  ic.generate_photon_energies(eph_PL1, g_PL1);
  ic.generate_photon_energies(eph_PL2, g_PL2);
  cudaDeviceSynchronize();
  // timer::show_duration_since_stamp("gen photon energies on gpu",
  // "ms"); timer::stamp();
  for (uint32_t i = 0; i < N_samples; i++) {
    eph_M[i] = ic.gen_photon_e(g_M[i]);
  }
  // timer::show_duration_since_stamp("gen photon energies on cpu",
  // "ms"); eph_M.copy_to_host();
  eph_PL1.copy_to_host();
  eph_PL2.copy_to_host();
  std::vector<Scalar> test_e_M(N_samples);
  std::vector<Scalar> test_e_PL1(N_samples);
  std::vector<Scalar> test_e_PL2(N_samples);
  for (uint32_t i = 0; i < N_samples; i++) {
    test_e_M[i] = eph_M[i];
    test_e_PL1[i] = eph_PL1[i];
    test_e_PL2[i] = eph_PL2[i];
  }
  // std::vector<std::vector<Scalar>> test_e1p(params.n_gamma);
  // std::vector<std::vector<Scalar>> test_e(params.n_gamma);
  // for (int n = 0; n < params.n_gamma; n++) {
  //   // test_e1p[n] = std::vector<Scalar>(N_samples);
  //   test_e[n] = std::vector<Scalar>(N_samples);
  //   for (uint32_t i = 0; i < N_samples; i++) {
  //     // Logger::print_info("at {}", i);
  //     // test_e1p[n][i] =
  //     ic.gen_e1p(ic.find_n_gamma(ic.gammas()[n]));
  //     // test_e1p[n][i] =
  //     ic.gen_ep(ic.find_n_gamma(ic.gammas()[n]), 1.5f); test_e[n][i]
  //     = ic.gen_photon_e(ic.gammas()[n]);
  //   }
  // }
  // DataSet data_teste1p = datafile.createDataSet<Scalar>(
  //     "test_e1p", DataSpace::From(test_e1p));
  // data_teste1p.write(test_e1p);
  DataSet data_e_M = datafile.createDataSet<Scalar>(
      "test_e_M", DataSpace::From(test_e_M));
  data_e_M.write(test_e_M);
  DataSet data_e_PL1 = datafile.createDataSet<Scalar>(
      "test_e_PL1", DataSpace::From(test_e_PL1));
  data_e_PL1.write(test_e_PL1);
  DataSet data_e_PL2 = datafile.createDataSet<Scalar>(
      "test_e_PL2", DataSpace::From(test_e_PL2));
  data_e_PL2.write(test_e_PL2);

  cu_array<Scalar> g_mono(N_samples);
  cu_array<Scalar> eph_mono(N_samples);
  std::vector<Scalar> test_mono(N_samples);
  for (int n = 1; n <= 5; n++) {
    g_mono.assign_dev(pow(10.0, n));
    ic.generate_photon_energies(eph_mono, g_mono);
    cudaDeviceSynchronize();
    eph_mono.copy_to_host();
    for (uint32_t i = 0; i < N_samples; i++) {
      test_mono[i] = eph_mono[i];
    }
    DataSet data_mono = datafile.createDataSet<Scalar>(
        fmt::format("mono1e{}", n), DataSpace::From(test_mono));
    data_mono.write(test_mono);
  }

  return 0;
}
