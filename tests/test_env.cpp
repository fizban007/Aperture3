#include <cstring>
#include "sim_environment.h"
#include "catch.hpp"

using namespace Aperture;
using namespace std;

// Defined in test_config.cpp
void check_test_params(const SimParams& params);

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
  // Aperture::cu_sim_environment &env = Aperture::cu_sim_environment::get_instance().initialize(&my_argc, &my_argv);
  // env = std::make_unique<Aperture::cu_sim_environment>(&my_argc, &my_argv);
  sim_environment env(&my_argc, &my_argv);

  CHECK(env.params().max_steps == 100000);
  CHECK(env.params().data_interval == 20);
  CHECK(env.params().conf_file == "test.toml");

  // check_test_params(env.params());
  // delete env;
  CHECK(env.mesh().dims[0] == 10006);
  CHECK(env.mesh().dims[1] == 5);
  CHECK(env.mesh().dims[2] == 11);

  SimParams p;
  p.data_dir = env.params().data_dir;

  // env.check_dev_params(p);
  // check_test_params(p);

  for (int i = 0; i < my_argc; i++) delete[] my_argv[i];
  delete[] my_argv;
}
