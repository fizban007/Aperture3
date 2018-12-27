#include "catch.hpp"
#include "data/fields.h"
#include "omp.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include <immintrin.h>
#include <iostream>
#include <random>
#include <vector>

using namespace Aperture;

TEST_CASE("avx gather on field", "[avx2]") {
  int N1 = 260, N2 = 290;
  Grid g(N1, N2);
  scalar_field<float> f(g);

  f.assign(2.0f);
  auto& data = f.data();

  scalar_field<float> f2(g);

  timer::stamp();
#if defined(__AVX2__)
  for (int j = 0; j < N2; j++) {
    for (int i = 2; i < N1 - 2; i += 8) {
      int offset = i * sizeof(float) + j * data.pitch();
      __m256i idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
      // __m256i off = _mm256_set1_epi32(offset);
      // __m256i size = _mm256_set1_epi32(sizeof(float));
      // idx = _mm256_add_epi32(idx, off);

      __m256 vf1 = _mm256_i32gather_ps(
          (float*)((char*)data.data() + offset), idx, 1);
      __m256 vf2 = _mm256_i32gather_ps(
          (float*)((char*)data.data() + offset + sizeof(float)), idx,
          1);

      __m256 x1 =
          _mm256_set_ps(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f);
      __m256 ones = _mm256_set1_ps(1.0f);
      __m256 x1s = _mm256_sub_ps(ones, x1);

      __m256 vf = _mm256_fmadd_ps(x1, vf2, _mm256_mul_ps(x1s, vf1));

      _mm256_storeu_ps((float*)((char*)f2.data().data() + offset), vf);
    }
  }
#endif
  timer::show_duration_since_stamp("vectorized linear interpolation",
                                   "us");
}

TEST_CASE("normal gather on field", "[avx2]") {
  int N1 = 260, N2 = 290, N3 = 260;
  Grid g(N1, N2, N3);
  scalar_field<float> f(g);
  auto& mesh = g.mesh();

  f.assign(2.0f);
  auto& data = f.data();

  uint32_t N = 1000000;
  std::vector<uint32_t> cells(N);
  std::vector<float> xs(N);
  std::vector<float> results(N);

  std::default_random_engine gen;
  std::uniform_int_distribution<uint32_t> dist(10, N1 - 10);
  std::uniform_real_distribution<float> dist_f(0.0, 1.0);

  for (uint32_t i = 0; i < N; i++) {
    cells[i] = mesh.get_idx(dist(gen), dist(gen), dist(gen));
    xs[i] = dist_f(gen);
  }

  timer::stamp();
// #pragma omp simd
  for (uint32_t i = 0; i < N; i++) {
    uint32_t c = cells[i];
    int c1 = mesh.get_c1(c);
    int c2 = mesh.get_c2(c);
    int c3 = mesh.get_c3(c);
    float x = xs[i];
    results[i] = (1.0f - x) * f(c1, c2) + x * f(c1 + 1, c2);
  }
  auto t = timer::get_duration_since_stamp("us");
  Logger::print_info(
      "Ordinary interpolation for {} particles took {}us.", N, t);
}
