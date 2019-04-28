#include <iostream>
#include "cu_sim_environment.h"

using namespace Aperture;

int main(int argc, char *argv[])
{
  cu_sim_environment env(&argc, &argv);

  std::cout << env.params().max_steps << std::endl;
  return 0;
}
