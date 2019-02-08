#include "cuda/radiation/rt_ic_impl.hpp"
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
  DataSet data_gammas = datafile.createDataSet<Scalar>(
      "gammas", DataSpace(ic.gammas().size()));
  data_gammas.write(ic.gammas().data());
  DataSet data_rates = datafile.createDataSet<Scalar>(
      "rates", DataSpace(ic.rate().size()));
  data_rates.write(ic.rate().data());
  DataSet data_ep = datafile.createDataSet<Scalar>(
      "ep", DataSpace(ic.ep().size()));
  data_ep.write(ic.ep().data());

  boost::multi_array<Scalar, 2> out_array;
  out_array.resize(boost::extents[params.n_gamma][params.n_ep]);
  for (int j = 0; j < params.n_gamma; j++) {
    for (int i = 0; i < params.n_ep; i++) {
      out_array[j][i] = ic.np()(i, j);
    }
  }
  DataSet data_np = datafile.createDataSet<Scalar>(
      "np", DataSpace::From(out_array));
  data_np.write(out_array);
  return 0;
}
