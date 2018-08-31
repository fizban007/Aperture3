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
  VectorField<Scalar> u, v, u_comp;
  ScalarField<Scalar> f;
  const Quadmesh& mesh;

 public:
  FiniteDiffTests() :
      env("test_diff.toml"),
      u(env.local_grid()),
      v(env.local_grid()),
      u_comp(env.local_grid()),
      f(env.local_grid()),
      mesh(env.local_grid().mesh()) {
    // Set everything to zero
    u.initialize();
    v.initialize();
    f.initialize();
    u_comp.initialize();
    u.sync_to_host();
    v.sync_to_host();
    f.sync_to_host();
    u_comp.sync_to_host();
    init_u();
  }

  void init_u() {
    // Initialize field components
    u.initialize(0, [](Scalar x1, Scalar x2, Scalar x3) {
                      return 3.0 * x2 * x2 * x3;
                    });
    u.initialize(1, [](Scalar x1, Scalar x2, Scalar x3) {
                      return 2.0 * x1 * x3 * x3;
                    });
    u.initialize(2, [](Scalar x1, Scalar x2, Scalar x3) {
                      return 1.0 * x1 * x2 * x3;
                    });
    u.sync_to_device();
  }

  void init_comp() {
    u_comp.initialize(0, [](Scalar x1, Scalar x2, Scalar x3) {
                           return - 3.0 * x1 * x3;
                         });
    u_comp.initialize(1, [](Scalar x1, Scalar x2, Scalar x3) {
                           return 3.0*x2*x2 - x2 * x3;
                         });
    u_comp.initialize(2, [](Scalar x1, Scalar x2, Scalar x3) {
                           return 2.0*x3*x3 - 6.0*x2*x3;
                         });
  }
};

TEST_CASE_METHOD(FiniteDiffTests, "Curl, stagger like E", "[FiniteDiff]") {
  int half = mesh.dims[0] / 2;
  v.initialize();
  CHECK(v(0, half, half, half) == 0.0f);
  u_comp.set_stagger(0, 0b110);
  u_comp.set_stagger(1, 0b101);
  u_comp.set_stagger(2, 0b011);
  init_comp();

  timer::stamp();
  // Compute the curl and add the result to v
  const int N = 20;
  for (int i = 0; i < N; i++)
    curl_add(v, u);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  auto time = timer::get_duration_since_stamp("ms") / (float)N;
  Logger::print_info("Curl took {}ms, overall bandwidth is {}GB/s", time,
                     mesh.size()*sizeof(Scalar)*6.0*1.0e-6/time);
  // timer::show_duration_since_stamp("Taking curl", "ms");
  v.sync_to_host();
  for (int k = 0; k < mesh.dims[2]; k+=4) {
    for (int j = 0; j < mesh.dims[1]; j+=4) {
      for (int i = 0; i < mesh.dims[0]; i+=4) {
        // std::cout << i << ", " << j << ", " << k << std::endl;
        if (!mesh.is_in_bulk(i, j, k)) {
          INFO(i << ", " << j << ", " << k);
          REQUIRE(v(2, i, j, k) == 0.0f);
        } else {
          REQUIRE(v(2, i, j, k)/(double)N == Approx((u(1, i + 1, j, k) - u(1, i, j, k))/mesh.delta[0]
                                          -(u(0, i, j + 1, k) - u(0, i, j, k))/mesh.delta[1]));
          REQUIRE(v(1, i, j, k)/(double)N == Approx((u(0, i, j, k + 1) - u(0, i, j, k))/mesh.delta[2]
                                                -(u(2, i + 1, j, k) - u(2, i, j, k))/mesh.delta[0]));
          REQUIRE(v(0, i, j, k)/(double)N == Approx((u(2, i, j + 1, k) - u(2, i, j, k))/mesh.delta[1]
                                                -(u(1, i, j, k + 1) - u(1, i, j, k))/mesh.delta[2]));
          // REQUIRE(v(2, i, j, k)/(double)N == Approx(u_comp(2, i, j, k)));
          // REQUIRE(v(1, i, j, k)/(double)N == Approx(u_comp(1, i, j, k)));
          // REQUIRE(v(0, i, j, k)/(double)N == Approx(u_comp(0, i, j, k)));
        }
      }
    }
  }
}

TEST_CASE_METHOD(FiniteDiffTests, "Curl, stagger like B", "[FiniteDiff]") {
  int half = mesh.dims[0] / 2;
  v.initialize();

  u.set_stagger(0, 0b110);
  u.set_stagger(1, 0b101);
  u.set_stagger(2, 0b011);
  init_u();
  // u.sync_to_host();
  // v.sync_to_host();
  CHECK(v(0, half, half, half) == 0.0f);
  u_comp.set_stagger(0, 0b001);
  u_comp.set_stagger(1, 0b010);
  u_comp.set_stagger(2, 0b100);
  init_comp();

  // cudaDeviceSynchronize();

  timer::stamp();
  // Compute the curl and add the result to v
  const int N = 20;
  for (int i = 0; i < N; i++)
    curl_add(v, u);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  auto time = timer::get_duration_since_stamp("ms") / (float)N;
  Logger::print_info("Curl took {}ms, overall bandwidth is {}GB/s", time,
                     mesh.size()*sizeof(Scalar)*6.0*1.0e-6/time);
  // timer::show_duration_since_stamp("Taking curl", "ms");
  v.sync_to_host();
  for (int k = 0; k < mesh.dims[2]; k+=4) {
    for (int j = 0; j < mesh.dims[1]; j+=4) {
      for (int i = 0; i < mesh.dims[0]; i+=4) {
        // std::cout << i << ", " << j << ", " << k << std::endl;
        if (!mesh.is_in_bulk(i, j, k)) {
          INFO(i << ", " << j << ", " << k);
          REQUIRE(v(2, i, j, k) == 0.0f);
        } else {
          INFO(i << ", " << j << ", " << k);
          // REQUIRE(v(2, i, j, k) == Approx(-1.0f*N));
          REQUIRE(v(2, i, j, k)/(double)N == Approx((u(1, i, j, k) - u(1, i - 1, j, k))/mesh.delta[0]
                                          -(u(0, i, j, k) - u(0, i, j - 1, k))/mesh.delta[1]));
          REQUIRE(v(1, i, j, k)/(double)N == Approx((u(0, i, j, k) - u(0, i, j, k - 1))/mesh.delta[2]
                                                -(u(2, i, j, k) - u(2, i - 1, j, k))/mesh.delta[0]));
          REQUIRE(v(0, i, j, k)/(double)N == Approx((u(2, i, j, k) - u(2, i, j - 1, k))/mesh.delta[1]
                                                -(u(1, i, j, k) - u(1, i, j, k - 1))/mesh.delta[2]));
          // REQUIRE(v(2, i, j, k)/(double)N == Approx(u_comp(2, i, j, k)));
          // REQUIRE(v(1, i, j, k)/(double)N == Approx(u_comp(1, i, j, k)));
          // REQUIRE(v(0, i, j, k)/(double)N == Approx(u_comp(0, i, j, k)));
        }
      }
    }
  }
}

TEST_CASE_METHOD(FiniteDiffTests, "Curl, old method", "[FiniteDiff]") {
  v.initialize();
  u.set_stagger(0, 0b110);
  u.set_stagger(1, 0b101);
  u.set_stagger(2, 0b011);
  init_u();

  // cudaDeviceSynchronize();

  timer::stamp();
  // Compute the curl and add the result to v
  const int N = 20;
  for (int i = 0; i < N; i++)
    curl(v, u);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  auto time = timer::get_duration_since_stamp("ms") / (float)N;
  Logger::print_info("Curl old method took {}ms, overall bandwidth is {}GB/s", time,
                     mesh.size()*sizeof(Scalar)*6.0*1.0e-6/time);
  // timer::show_duration_since_stamp("Taking curl", "ms");
  v.sync_to_host();
  for (int k = 0; k < mesh.dims[2]; k+=4) {
    for (int j = 0; j < mesh.dims[1]; j+=4) {
      for (int i = 0; i < mesh.dims[0]; i+=4) {
        // std::cout << i << ", " << j << ", " << k << std::endl;
        if (!mesh.is_in_bulk(i, j, k)) {
          INFO(i << ", " << j << ", " << k);
          REQUIRE(v(2, i, j, k) == 0.0f);
        } else {
          INFO(i << ", " << j << ", " << k);
          // REQUIRE(v(2, i, j, k) == Approx(-1.0f*N));
          REQUIRE(v(2, i, j, k) == Approx((u(1, i, j, k) - u(1, i - 1, j, k))/mesh.delta[0]
                                          -(u(0, i, j, k) - u(0, i, j - 1, k))/mesh.delta[1]));
          REQUIRE(v(1, i, j, k) == Approx((u(0, i, j, k) - u(0, i, j, k - 1))/mesh.delta[2]
                                                -(u(2, i, j, k) - u(2, i - 1, j, k))/mesh.delta[0]));
          REQUIRE(v(0, i, j, k) == Approx((u(2, i, j, k) - u(2, i, j - 1, k))/mesh.delta[1]
                                                -(u(1, i, j, k) - u(1, i, j, k - 1))/mesh.delta[2]));
        }
      }
    }
  }
}

TEST_CASE_METHOD(FiniteDiffTests, "Div", "[FiniteDiff]") {
  u.set_stagger(0, 0b001);
  u.set_stagger(1, 0b010);
  u.set_stagger(2, 0b100);

  // Initialize field components
  u.initialize(0, [](Scalar x1, Scalar x2, Scalar x3) {
                    return 4.0f * x1;
                  });
  u.initialize(1, [](Scalar x1, Scalar x2, Scalar x3) {
                    return 2.0f * x2;
                  });
  u.initialize(2, [](Scalar x1, Scalar x2, Scalar x3) {
                    return 0.0;
                  });
  u.sync_to_device();

  timer::stamp();
  // Compute the curl and add the result to v
  const int N = 20;
  for (int i = 0; i < N; i++)
    div(f, u);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  auto time = timer::get_duration_since_stamp("ms") / (float)N;
  Logger::print_info("Div took {}ms, overall bandwidth is {}GB/s", time,
                     mesh.size()*sizeof(Scalar)*4.0*1.0e-6/time);
  f.sync_to_host();

  for (int k = 0; k < mesh.dims[2]; k+=4) {
    for (int j = 0; j < mesh.dims[1]; j+=4) {
      for (int i = 0; i < mesh.dims[0]; i+=4) {
        if (!mesh.is_in_bulk(i, j, k)) {
          INFO(i << ", " << j << ", " << k);
          REQUIRE(f(i, j, k) == 0.0f);
        } else {
          INFO(i << ", " << j << ", " << k);
          REQUIRE(f(i, j, k) == 6.0f);
          // REQUIRE(f(i, j, k)/(double)N == Approx((u(0, i, j, k) - u(0, i - 1, j, k))/mesh.delta[0]
          //                                        +(u(1, i, j, k) - u(1, i, j - 1, k))/mesh.delta[1]));
        }
      }
    }
  }
}
