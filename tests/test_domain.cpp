#include "catch.hpp"
#include "cuda/core/cu_sim_data.h"
#include "cuda/core/sim_environment_dev.h"
#include <cstring>

using namespace Aperture;
using namespace std;

TEST_CASE("Making domain", "[Domain]") {
  int my_argc = 3;
  char** my_argv = new char*[my_argc + 1];
  for (int i = 0; i < my_argc; i++) my_argv[i] = new char[100];
  strcpy(my_argv[0], "aperture");
  strcpy(my_argv[1], "-c");
  strcpy(my_argv[2], "test_domain.toml");
  my_argv[3] = nullptr;

  cu_sim_environment env(&my_argc, &my_argv);

  cu_sim_data data(env);

  Logger::print_info("ptr size is {}, {}, {}",
                     data.E[0].grid().extent().width(),
                     data.E[0].grid().extent().height(),
                     data.E[0].grid().extent().depth());
}
