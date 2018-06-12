#include <iostream>
#include "sim_environment.h"

using namespace Aperture;

int main(int argc, char *argv[])
{
  Environment env(&argc, &argv);

  std::cout << env.params().max_steps << std::endl;
  return 0;
}
