#include "config_file.h"
#include "catch.hpp"

using namespace Aperture;

ConfigFile config;

void check_test_params(const SimParams& params) {
  CHECK(params.delta_t == Approx(0.3e-4));
  CHECK(params.q_e == Approx(0.1));
  CHECK(params.max_ptc_number == 100000000);
  CHECK(params.max_photon_number == 100000000);
  CHECK(params.create_pairs == true);
  CHECK(params.trace_photons == true);
  CHECK(params.gamma_thr == Approx(5.0));
  CHECK(params.photon_path == Approx(100.0));
  CHECK(params.ic_path == Approx(20.0));
  CHECK(params.spectral_alpha == Approx(1.2));
  CHECK(params.e_min == Approx(1.0e-5));
  CHECK(params.e_s == Approx(0.2));
  CHECK(params.track_percent == Approx(0.1));
  CHECK(params.data_dir == "/home/alex/storage/Data/1DpicCuda/");
  CHECK(params.periodic_boundary[0] == true);
  CHECK(params.periodic_boundary[1] == true);
  CHECK(params.periodic_boundary[2] == false);

  CHECK(params.N[0] == 10000);
  CHECK(params.N[1] == 3);
  CHECK(params.N[2] == 9);
  CHECK(params.guard[0] == 3);
  CHECK(params.guard[1] == 1);
  CHECK(params.guard[2] == 1);
  CHECK(params.lower[0] == Approx(1.0));
  CHECK(params.lower[1] == Approx(2.0));
  CHECK(params.lower[2] == Approx(3.0));
  CHECK(params.size[0] == Approx(100.0));
  CHECK(params.size[1] == Approx(10.0));
  CHECK(params.size[2] == Approx(2.0));
  CHECK(params.tile_size[0] == 64);
  CHECK(params.tile_size[1] == 1);
  CHECK(params.tile_size[2] == 1);
}

TEST_CASE("Simple parsing", "[config]") {
  SimParams params;
  CHECK(params.periodic_boundary[0] == false);
  CHECK(params.periodic_boundary[1] == false);
  CHECK(params.periodic_boundary[2] == false);

  // config.parse_file("test.toml", params);

  // check_test_params(params);
}
