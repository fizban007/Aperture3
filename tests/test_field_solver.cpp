#include "core/grid.h"
#include "data/fields.h"
#include "core/field_solver_default.h"
#include "utils/timer.h"
#include "utils/logger.h"
#include "catch.hpp"

using namespace Aperture;

TEST_CASE("Using the default 3d field solver", "[field_solver]") {
  Grid g(66, 66, 66);
  auto& mesh = g.mesh();

  mesh.guard[0] = mesh.guard[1] = mesh.guard[2] = 1;
  mesh.sizes[0] = mesh.sizes[1] = mesh.sizes[2] = 1.0f;
  mesh.lower[0] = mesh.lower[1] = mesh.lower[2] = 0.0f;
  for (int i = 0; i < 3; i++) {
    mesh.delta[i] = mesh.sizes[i] / mesh.reduced_dim(i);
    mesh.inv_delta[i] = 1.0 / mesh.delta[i];
  }
  mesh.dimension = mesh.dim();
  
  vector_field<float> e(g);
  vector_field<float> b(g);

  e.initialize(0, [](Scalar x1, Scalar x2, Scalar x3){
                    return 2.0f * x1 + 3.0f * x2 + 1.5f * x3 * x3;
                  });
  e.initialize(1, [](Scalar x1, Scalar x2, Scalar x3){
                    return 3.0f * x1 + 4.0f * x2 + x3;
                  });

  field_solver_default solver(g);

  timer::stamp();
  solver.compute_B_update(b, e, 0.01);

  auto t = timer::get_duration_since_stamp("ms");
  Logger::print_info("Update B took {}ms for {}x{}x{}", t, 66, 66, 66);
}
