#include "cuda/radiation/rt_ic.h"
#include "radiation/spectra.h"
#include "sim_params.h"
#include "utils/logger.h"
#include <boost/math/quadrature/gauss.hpp>
#include <cmath>
#include <iostream>
#include <vector>

#include <boost/multi_array.hpp>
#define H5_USE_BOOST
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

using namespace Aperture;
using namespace HighFive;

Scalar e_min = 1.0e-6;
Scalar e_max = 1.0e3;

int
main(int argc, char *argv[]) {
  SimParams params;
  params.n_gamma = 200;
  params.n_ep = 4000;
  inverse_compton ic(params);

  // Spectra::power_law_hard ne(0.2, e_min, e_max);
  // Spectra::power_law_soft ne(1.2, e_min, e_max);
  Spectra::black_body ne(0.001);
  ic.init(ne, ne.emin(), ne.emax());

  File datafile("spectrum.h5",
                File::ReadWrite | File::Create | File::Truncate);
  Logger::print_info("gammas size is {}", ic.gammas().size());
  DataSet data_gammas = datafile.createDataSet<Scalar>(
      "gammas", DataSpace(ic.gammas().size()));
  data_gammas.write(ic.gammas().data());
  DataSet data_rates = datafile.createDataSet<Scalar>(
      "rates", DataSpace(ic.rate().size()));
  data_rates.write(ic.rate().data());
  DataSet data_ep =
      datafile.createDataSet<Scalar>("ep", DataSpace(ic.ep().size()));
  data_ep.write(ic.ep().data());

  boost::multi_array<Scalar, 2> out_array;
  out_array.resize(boost::extents[params.n_gamma][params.n_ep]);
  for (int j = 0; j < params.n_gamma; j++) {
    for (int i = 0; i < params.n_ep; i++) {
      out_array[j][i] = ic.np()(i, j);
    }
  }
  DataSet data_np =
      datafile.createDataSet<Scalar>("np", DataSpace::From(out_array));
  data_np.write(out_array);

  for (int j = 0; j < params.n_gamma; j++) {
    for (int i = 0; i < params.n_ep; i++) {
      out_array[j][i] = ic.dnde1p()(i, j);
    }
  }
  DataSet data_dnde1p = datafile.createDataSet<Scalar>(
      "dnde1p", DataSpace::From(out_array));
  data_dnde1p.write(out_array);

  const uint32_t N_samples = 100000;
  std::vector<std::vector<Scalar>> test_e1p(params.n_gamma);
  std::vector<std::vector<Scalar>> test_e(params.n_gamma);
  for (int n = 0; n < params.n_gamma; n++) {
    test_e1p[n] = std::vector<Scalar>(N_samples);
    test_e[n] = std::vector<Scalar>(N_samples);
    for (uint32_t i = 0; i < N_samples; i++) {
      // Logger::print_info("at {}", i);
      test_e1p[n][i] = ic.gen_e1p(ic.find_n_gamma(ic.gammas()[n]));
      test_e[n][i] = ic.gen_photon_e(ic.gammas()[n]);
    }
  }
  DataSet data_teste1p = datafile.createDataSet<Scalar>(
      "test_e1p", DataSpace::From(test_e1p));
  data_teste1p.write(test_e1p);
  DataSet data_teste = datafile.createDataSet<Scalar>(
      "test_e", DataSpace::From(test_e));
  data_teste.write(test_e);
  return 0;
}
