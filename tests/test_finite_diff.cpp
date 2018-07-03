#include "data/particles.h"
#include "utils/timer.h"
#include "utils/logger.h"
#include "cuda/constant_mem_func.h"
#include "algorithms/finite_diff.h"
#include "sim_environment.h"
#include "cuda_runtime.h"
#include "catch.hpp"

using namespace Aperture;

class FiniteDiffTests {
 protected:
  Environment env;
  VectorField<Scalar> u, v;
  ScalarField<Scalar> f;
  const Quadmesh& mesh;

 public:
  FiniteDiffTests() :
      env("test_diff.toml"),
      u(env.local_grid()),
      v(env.local_grid()),
      f(env.local_grid()),
      mesh(env.local_grid().mesh()) {
    // Set everything to zero
    u.initialize();
    v.initialize();
    f.initialize();
    u.sync_to_host();
    v.sync_to_host();
    f.sync_to_host();
  }
};

TEST_CASE_METHOD(FiniteDiffTests, "Curl", "[FiniteDiff]") {
  int half = mesh.dims[0] / 2;

  // u.sync_to_host();
  // v.sync_to_host();
  CHECK(u(0, half, half, half) == 0.0f);
  CHECK(v(0, half, half, half) == 0.0f);

  // Initialize field components
  u.initialize(0, [](Scalar x1, Scalar x2, Scalar x3) {
                    return 3.0 * x2;
                  });
  u.initialize(1, [](Scalar x1, Scalar x2, Scalar x3) {
                    return 2.0 * x1;
                  });
  Scalar d1du2 = (u(1, half, half, half) - u(1, half-1, half, half)) / mesh.delta[0];
  Scalar d2du1 = (u(0, half, half, half) - u(0, half, half-1, half)) / mesh.delta[1];
  CHECK(d1du2 - d2du1 == Approx(-1.0f));
  u.sync_to_device();
  // // cudaDeviceSynchronize();

  timer::stamp();
  // Compute the curl and add the result to v
  const int N = 100;
  for (int i = 0; i < N; i++)
    curl(v, u);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  auto time = timer::get_duration_since_stamp("ms") / 100.0;
  Logger::print_info("Curl took {}ms, overall bandwidth is {}GB/s", time,
                     mesh.size()*sizeof(Scalar)*2.0*1.0e-6/time);
  // timer::show_duration_since_stamp("Taking curl", "ms");
  // v.sync_to_host();
  // for (int k = 0; k < mesh.dims[2]; k++) {
  //   for (int j = 0; j < mesh.dims[1]; j++) {
  //     for (int i = 0; i < mesh.dims[0]; i++) {
  //       // std::cout << i << ", " << j << ", " << k << std::endl;
  //       if (i == 0 || i == mesh.dims[0] - 1 ||
  //           j == 0 || j == mesh.dims[1] - 1 ||
  //           k == 0 || k == mesh.dims[2] - 1)
  //         REQUIRE(v(2, i, j, k) == 0.0f);
  //       else
  //         REQUIRE(v(2, i, j, k) == Approx(-1.0f));
  //     }
  //   }
  // }
}

TEST_CASE_METHOD(FiniteDiffTests, "Div", "[FiniteDiff]") {
  // Initialize field components
  u.initialize(0, [](Scalar x1, Scalar x2, Scalar x3) {
                    return 3.0 * x1;
                  });
  u.initialize(1, [](Scalar x1, Scalar x2, Scalar x3) {
                    return 2.0 * x2;
                  });
  u.sync_to_device();

  timer::stamp();
  // Compute the curl and add the result to v
  div(f, u);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  timer::show_duration_since_stamp("Taking div", "ms");
  f.sync_to_host();

  for (int k = 0; k < mesh.dims[2]; k++) {
    for (int j = 0; j < mesh.dims[1]; j++) {
      for (int i = 0; i < mesh.dims[0]; i++) {
        if (i == 0 || i == mesh.dims[0] - 1 ||
            j == 0 || j == mesh.dims[1] - 1 ||
            k == 0 || k == mesh.dims[2] - 1)
          REQUIRE(f(i, j, k) == 0.0f);
        else
          REQUIRE(f(i, j, k) == Approx(5.0f));
      }
    }
  }
}
