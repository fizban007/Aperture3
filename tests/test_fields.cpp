#include "data/fields.h"
#include "utils/timer.h"
#include "utils/logger.h"
#include "catch.hpp"

using namespace Aperture;

TEST_CASE("Scalar Field multiplyBy", "[scalar_field]") {
  int N1 = 260, N2 = 290;
  Grid g(N1, N2);
  scalar_field<float> f(g);

  
  REQUIRE(f(100, 100) == 0.0f);

  f.assign(2.0f);
  f.multiplyBy(3.0f);

  REQUIRE(f(100, 100) == 6.0f);

  scalar_field<float> f2(g);

  f2.assign(3.0f);
  f.multiplyBy(f2);

  for (int j = 0; j < N2; j++) {
    for (int i = 0; i < N1; i++) {
      REQUIRE(f(i, j) == 18.0f);
    }
  }
}

TEST_CASE("Performance of multiplyBy", "[scalar_field, performance]") {
  int N1 = 260, N2 = 290, N3 = 260;
  Grid g(N1, N2, N3);
  scalar_field<float> f1(g), f2(g);

  f1.assign(2.0f);
  f2.assign(3.0f);

  timer::stamp();
  f1.multiplyBy(f2);
  auto t = timer::get_duration_since_stamp("ms");
  Logger::print_info("multiplyBy took {}ms for {}x{}x{}", t, N1, N2, N3);

  timer::stamp();
  f1.multiplyBy_slow(f2);
  t = timer::get_duration_since_stamp("ms");
  Logger::print_info("multiplyBy without simd took {}ms for {}x{}x{}", t, N1, N2, N3);
}

