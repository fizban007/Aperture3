#include "catch.hpp"
#include "core/multi_array.h"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace Aperture;

TEST_CASE("Initialize and Using MultiArray", "[MultiArray]") {
  int X = 260, Y = 310;
  Extent ext(X, Y);

  multi_array<float> a(ext);

  REQUIRE(a.width() == X);
  REQUIRE(a.height() == Y);
  REQUIRE(a.depth() == 1);
  REQUIRE(a.size() == 260 * 310);

  a.assign(2.0f);

  // Subscripting works
  for (int j = 0; j < Y; j++) {
    for (int i = 0; i < X; i++) {
      REQUIRE(a(i, j) == 2.0f);
    }
  }
}

TEST_CASE("Using offset directly", "[MultiArray]") {
  int N1 = 260, N2 = 310;

  multi_array<float> a(N1, N2);
  a.assign(1.0f);

  a(100, 200) = 3.0f;

  REQUIRE(a[100 + 200 * N1] == 3.0f);
}

TEST_CASE("Copying a subset", "[MultiArray]") {
  int N1 = 260, N2 = 300;
  int n1 = 10, n2 = 30;

  multi_array<float> a(N1, N2);
  a.assign(1.0f);

  multi_array<float> b(n1, n2);
  b.assign(2.0f);

  a.copy_from(b, Index(0, 0), Index(20, 50), Extent(n1, n2));

  for (int j = 0; j < N2; j++) {
    for (int i = 0; i < N1; i++) {
      if (i >= 20 && i < 20 + n1 && j >= 50 && j < 50 + n2)
        REQUIRE(a(i, j) == 2.0f);
      else
        REQUIRE(a(i, j) == 1.0f);
    }
  }
}

#ifdef USE_CUDA
TEST_CASE("Copying a subset, over device", "[MultiArray]") {
  int N1 = 260, N2 = 300;
  int n1 = 10, n2 = 30;

  multi_array<float> a(N1, N2);
  a.assign_dev(1.0f);

  multi_array<float> b(n1, n2);
  b.assign_dev(2.0f);

  a.copy_from(b, Index(0, 0), Index(20, 50), Extent(n1, n2),
              (int)cudaMemcpyDeviceToDevice);
  a.sync_to_host();

  for (int j = 0; j < N2; j++) {
    for (int i = 0; i < N1; i++) {
      if (i >= 20 && i < 20 + n1 && j >= 50 && j < 50 + n2)
        REQUIRE(a(i, j) == 2.0f);
      else
        REQUIRE(a(i, j) == 1.0f);
    }
  }
}
#endif
