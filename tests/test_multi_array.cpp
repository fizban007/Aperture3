#include "data/multi_array.h"
#include "catch.hpp"

using namespace Aperture;

TEST_CASE("Initialize and Using MultiArray", "[MultiArray]") {
  int X = 260, Y = 310;
  Extent ext(X, Y);
  
  multi_array<float> a(ext);

  REQUIRE(a.width() == X);
  REQUIRE(a.height() == Y);
  REQUIRE(a.depth() == 1);
  REQUIRE(a.pitch() == 1088);
  REQUIRE(a.size() == 1088 * 310);

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

  REQUIRE(a[100 * sizeof(float) + 200 * a.pitch()] == 3.0f);
}
