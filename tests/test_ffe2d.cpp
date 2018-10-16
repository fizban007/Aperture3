#include <memory>
#include "utils/timer.h"
#include "utils/logger.h"
#include "cuda/constant_mem_func.h"
#include "algorithms/field_solver_ffe_cyl.h"
#include "catch.hpp"

using namespace Aperture;

class FFETests {
 protected:
  Environment env;
  VectorField<Scalar> E, B;
  VectorField<Scalar> E_out, B_out, J_out;
      // , u_comp;
  // ScalarField<Scalar> f;
  const Quadmesh& mesh;
  FieldSolver_FFE_Cyl solver;

 public:
  FFETests() :
      env("test_diff.toml"),
      E(env.local_grid()),
      B(env.local_grid()),
      E_out(env.local_grid()),
      B_out(env.local_grid()),
      J_out(env.local_grid()),
      mesh(env.local_grid().mesh()),
      solver(env.local_grid()) {
  }

  void init_u() {
    // Initialize field components
    E.initialize(0, [](Scalar x1, Scalar x2, Scalar x3) {
                      return 3.0 * x2 * x2 * x3;
                    });
    E.initialize(1, [](Scalar x1, Scalar x2, Scalar x3) {
                      return 2.0 * x1 * x3 * x3;
                    });
    E.initialize(2, [](Scalar x1, Scalar x2, Scalar x3) {
                      return 1.0 * x1 * x2 * x3;
                    });
    E.sync_to_device();
    B.initialize(0, [](Scalar x1, Scalar x2, Scalar x3) {
                      return 3.0 * x2 * x2 * x3;
                    });
    B.initialize(1, [](Scalar x1, Scalar x2, Scalar x3) {
                      return 2.0 * x1 * x3 * x3;
                    });
    B.initialize(2, [](Scalar x1, Scalar x2, Scalar x3) {
                      return 1.0 * x1 * x2 * x3;
                    });
    B.sync_to_device();
  }
};

TEST_CASE_METHOD(FFETests, "FF Cylindrical substep", "[FFE]") {
  init_u();

  timer::stamp("FFE");
  solver.update_field_substep(E_out, B_out, J_out, E, B, 0.01);
  cudaDeviceSynchronize();
  timer::show_duration_since_stamp("FFE cylindrical substep", "ms", "FFE");
}
