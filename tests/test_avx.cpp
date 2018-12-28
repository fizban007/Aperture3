#include "catch.hpp"
#include "data/fields.h"
#include "omp.h"
#include "utils/logger.h"
#include "utils/simd.h"
#include "utils/timer.h"
#include <immintrin.h>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>

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

TEST_CASE("avx interpolation on field", "[avx2]") {
  int N1 = 260, N2 = 290, N3 = 260;
  Grid g(N1, N2, N3);
  scalar_field<float> f(g);
  auto& mesh = g.mesh();

  f.assign(2.0f);
  auto& data = f.data();

  uint32_t N = 5000000;
  std::vector<uint32_t> cells(N);
  std::vector<float> x1v(N);
  std::vector<float> x2v(N);
  std::vector<float> x3v(N);
  std::vector<float> results1(N);
  std::vector<float> results2(N);

  std::default_random_engine gen;
  std::uniform_int_distribution<uint32_t> dist(10, N1 - 10);
  std::uniform_real_distribution<float> dist_f(0.0, 1.0);

  for (uint32_t i = 0; i < N; i++) {
    cells[i] = mesh.get_idx(dist(gen), dist(gen), dist(gen));
    x1v[i] = dist_f(gen);
    x2v[i] = dist_f(gen);
    x3v[i] = dist_f(gen);
  }
  std::sort(cells.begin(), cells.end());

  timer::stamp();
  for (uint32_t i = 0; i < N; i += 8) {
    Vec8ui c;
    c.load(cells.data() + i);
    Vec8ui d = c / Divisor_ui(data.width());
    Vec8ui c1s = c - d * data.width();
    Vec8ui offsets = c1s * sizeof(float) + d * data.pitch();

    Vec8f x1; x1.load(x1v.data() + i);
    Vec8f x2; x2.load(x2v.data() + i);
    Vec8f x3; x3.load(x3v.data() + i);

    uint32_t k_off = data.pitch() * data.height();
    Vec8f f000 = _mm256_i32gather_ps((float*)data.data(), offsets, 1);
    Vec8f f001 = _mm256_i32gather_ps((float*)data.data(), offsets + sizeof(float), 1);
    Vec8f f010 = _mm256_i32gather_ps((float*)data.data(), offsets + data.pitch(), 1);
    Vec8f f011 = _mm256_i32gather_ps((float*)data.data(), offsets + (sizeof(float) + data.pitch()), 1);
    Vec8f f100 = _mm256_i32gather_ps((float*)data.data(), offsets + k_off, 1);
    Vec8f f101 = _mm256_i32gather_ps((float*)data.data(), offsets + (k_off + sizeof(float)), 1);
    Vec8f f110 = _mm256_i32gather_ps((float*)data.data(), offsets + (k_off + data.pitch()), 1);
    Vec8f f111 = _mm256_i32gather_ps((float*)data.data(), offsets + (k_off + sizeof(float) + data.pitch()), 1);

    f000 = simd::lerp(x3, f000, f100);
    f010 = simd::lerp(x3, f010, f110);
    f001 = simd::lerp(x3, f001, f101);
    f011 = simd::lerp(x3, f011, f111);

    f000 = simd::lerp(x2, f000, f010);
    f001 = simd::lerp(x2, f001, f011);

    f000 = simd::lerp(x1, f000, f001);
    f000.store(results1.data() + i);
  }
  auto t = timer::get_duration_since_stamp("us");
  Logger::print_info(
      "explicit AVX interpolation for {} particles took {}us.", N, t);

  timer::stamp();
// #pragma omp simd
  for (uint32_t i = 0; i < N; i++) {
    uint32_t c = cells[i];
    size_t offset = data.get_offset(c);
    size_t k_off = data.pitch() * data.height();
    float x1 = x1v[i];
    float x2 = x2v[i];
    float x3 = x3v[i];
    float nx1 = 1.0f - x1;
    float nx2 = 1.0f - x2;
    float nx3 = 1.0f - x3;
    results2[i] = nx1 * nx2 * nx3 * data[offset]
        + x1 * nx2 * nx3 * data[offset + sizeof(float)]
        + nx1 * x2 * nx3 * data[offset + data.pitch()]
        + nx1 * nx2 * x3 * data[offset + k_off]
        + x1 * x2 * nx3 * data[offset + sizeof(float) + data.pitch()]
        + x1 * nx2 * x3 * data[offset + sizeof(float) + k_off]
        + nx1 * x2 * x3 * data[offset + data.pitch() + k_off]
        + x1 * x2 * x3 * data[offset + sizeof(float) + data.pitch() + k_off];
  }
  t = timer::get_duration_since_stamp("us");
  Logger::print_info(
      "Ordinary interpolation for {} particles took {}us.", N, t);

  for (uint32_t i = 0; i < N; i++) {
    REQUIRE(results1[i] == Approx(results2[i]));
  }
}

TEST_CASE("Vec8ui division", "[vectorclass]") {
  Vec8ui v(80, 90, 100, 110, 120, 131, 142, 153);
  Vec8ui result = v / Divisor_ui(8);

  REQUIRE(result[0] == 10);
  REQUIRE(result[1] == 11);
  REQUIRE(result[2] == 12);
  REQUIRE(result[3] == 13);
  REQUIRE(result[4] == 15);
  REQUIRE(result[5] == 16);
  REQUIRE(result[6] == 17);
  REQUIRE(result[7] == 19);
}

TEST_CASE("i32gather_ps", "[avx2]") {
  int N1 = 10, N2 = 8, N3 = 0;
  Grid g(N1, N2, N3);
  scalar_field<float> f(g);
  auto& mesh = g.mesh();

  for (int j = 0; j < N2; j++) {
    for (int i = 0; i < N1; i++) {
      f(i, j) = i * j;
    }
  }
  auto& data = f.data();

  for (int j = 0; j < N2; j++) {
    Vec8ui c1s(0, 1, 2, 3, 4, 5, 6, 7);
    Vec8ui offsets = c1s * sizeof(float) + j * data.pitch();
    Vec8f f000 = _mm256_i32gather_ps((float*)data.data(), offsets, 1);
    for (int i = 0; i < 8; i++) {
      std::cout << f000[i] << " ";
    }
    std::cout << "\n";
  }
  
}
