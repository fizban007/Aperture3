#include <cstring>
#include "sim_environment.h"
#include "catch.hpp"

using namespace Aperture;
using namespace std;

TEST_CASE("Loading environment", "[Env]") {
  int my_argc = 7;
  char** my_argv = new char*[my_argc + 1];
  for (int i = 0; i < my_argc; i++) my_argv[i] = new char[100];
  strcpy(my_argv[0], "aperture");
  strcpy(my_argv[1], "-c");
  strcpy(my_argv[2], "test.toml");
  strcpy(my_argv[3], "-s");
  strcpy(my_argv[4], "100000");
  strcpy(my_argv[5], "-d");
  strcpy(my_argv[6], "20");
  my_argv[7] = nullptr;
  // Aperture::Environment &env = Aperture::Environment::get_instance().initialize(&my_argc, &my_argv);
  // env = std::make_unique<Aperture::Environment>(&my_argc, &my_argv);
  Environment env(&my_argc, &my_argv);

  CHECK(env.params().max_steps == 100000);
  CHECK(env.params().data_interval == 20);
  CHECK(env.params().conf_file == "test.toml");

  auto& params = env.params();
  CHECK(params.delta_t == Approx(0.3));
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
  CHECK(params.tile_size == 64);
  // delete env;
}
