#include "cuda/radiation/rt_ic_impl.hpp"
#include <boost/math/quadrature/gauss.hpp>
#include <cmath>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <iostream>
#include <vector>

using namespace Aperture;
using namespace HighFive;

Scalar e_min = 1.0e-18;
Scalar e_max = 1.0e-3;

struct Ne {
  Ne(Scalar delta) : delta_(delta) {}

  Scalar operator()(Scalar e) const {
    return pow(e / e_max, delta_) / e;
  }

  Scalar delta_;
};

int
main(int argc, char *argv[]) {
  SimParams params;
  params.n_gamma = 100;
  params.n_ep = 5000;
  inverse_compton ic(params);

  Ne ne(2.5);
  ic.init(ne, e_min, e_max);

  File datafile("spectrum.h5",
                File::ReadWrite | File::Create | File::Truncate);
  Logger::print_info("gammas size is {}", ic.gammas().size());
  DataSet data_gammas = datafile.createDataSet<Scalar>("gammas", DataSpace(
      ic.gammas().size()));
  data_gammas.write(ic.gammas().data());
  DataSet data_rates = datafile.createDataSet<Scalar>("rates", DataSpace(
      ic.rate().size()));
  data_rates.write(ic.rate().data());
  DataSet data_np = datafile.createDataSet<Scalar>("np", DataSpace(
      params.n_gamma, params.n_ep));
  return 0;
}
