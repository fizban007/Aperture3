#include "catch.hpp"
#include "core/interpolation.h"
#include "core/stagger.h"
#include "data/fields.h"
#include "omp.h"
#include "utils/avx_interp.hpp"
#include "utils/logger.h"
#include "utils/simd.h"
#include "utils/timer.h"
#include <algorithm>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <vector>

using namespace Aperture;

#if defined(__AVX2__)

TEST_CASE("avx gather on field", "[avx2]") {
  int N1 = 260, N2 = 290;
  Grid g(N1, N2);
  scalar_field<float> f(g);

  f.assign(2.0f);
  auto& data = f.data();

  scalar_field<float> f2(g);

  timer::stamp();
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
  timer::show_duration_since_stamp("vectorized linear interpolation",
                                   "us");
}

TEST_CASE("avx interpolation on field", "[avx2]") {
  Logger::print_info("Instruction set is {}", INSTRSET);

  int N1 = 260, N2 = 290, N3 = 260;
  Grid g(N1, N2, N3);
  scalar_field<float> f(g);
  auto& mesh = g.mesh();

  auto& data = f.data();

  uint32_t N = 5000000;
  std::vector<uint32_t> cells(N);
  std::vector<float> x1v(N);
  std::vector<float> x2v(N);
  std::vector<float> x3v(N);
  std::vector<float> results1(N);
  std::vector<float> results2(N);

  Stagger st(0b011);

  std::default_random_engine gen;
  std::uniform_int_distribution<uint32_t> dist(10, N1 - 10);
  std::uniform_real_distribution<float> dist_f(0.0, 1.0);

  f.initialize([&dist_f, &gen](float x1, float x2, float x3) {
    return dist_f(gen);
  });
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
    Vec8f x1;
    x1.load(x1v.data() + i);
    Vec8f x2;
    x2.load(x2v.data() + i);
    Vec8f x3;
    x3.load(x3v.data() + i);

    Vec8ib empty_mask = (c != Vec8ui(MAX_CELL));
    Vec8ui d = select(~empty_mask, Vec8ui(1 + mesh.dims[1]),
                      c / Divisor_ui(mesh.dims[0]));
    Vec8ui c1s = select(~empty_mask, Vec8ui(1), c - d * mesh.dims[0]);
    Vec8ui offsets = c1s * sizeof(float) + d * data.pitch();
    Vec8f f000 =
        interpolate_3d(data, offsets, x1, x2, x3, Stagger(0b111));

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
    results2[i] =
        nx1 * nx2 * nx3 *
            data[offset - sizeof(float) - data.pitch() - k_off] +
        x1 * nx2 * nx3 * data[offset - data.pitch() - k_off] +
        nx1 * x2 * nx3 * data[offset - sizeof(float) - k_off] +
        nx1 * nx2 * x3 * data[offset - sizeof(float) - data.pitch()] +
        x1 * x2 * nx3 * data[offset - k_off] +
        x1 * nx2 * x3 * data[offset - data.pitch()] +
        nx1 * x2 * x3 * data[offset - sizeof(float)] +
        x1 * x2 * x3 * data[offset];
  }

  t = timer::get_duration_since_stamp("us");
  Logger::print_info(
      "Ordinary interpolation for {} particles took {}us.", N, t);

  for (uint32_t i = 0; i < N; i++) {
    REQUIRE(results1[i] == Approx(results2[i]));
  }

#ifdef __AVX512F__
  timer::stamp();
  for (uint32_t i = 0; i < N; i += 16) {
    Vec16ui c;
    c.load(cells.data() + i);
    Vec16f x1;
    x1.load(x1v.data() + i);
    Vec16f x2;
    x2.load(x2v.data() + i);
    Vec16f x3;
    x3.load(x3v.data() + i);

    Vec16ib empty_mask = (c != Vec16ui(MAX_CELL));
    Vec16ui d = select(~empty_mask, Vec16ui(1 + mesh.dims[1]),
                       c / Divisor_ui(mesh.dims[0]));
    Vec16ui c1s = select(~empty_mask, Vec16ui(1), c - d * mesh.dims[0]);
    Vec16ui offsets = c1s * sizeof(float) + d * data.pitch();
    Vec16f f000 =
        interpolate_3d(data, offsets, x1, x2, x3, Stagger(0b111));

    f000.store(results1.data() + i);
  }
  t = timer::get_duration_since_stamp("us");
  Logger::print_info("AVX512 interpolation for {} particles took {}us.",
                     N, t);
#endif
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
    Vec8ui c1s(0, 0, 2, 2, 4, 4, 6, 6);
    Vec8ui offsets = c1s * sizeof(float) + j * data.pitch();
    Vec8f f000 = _mm256_i32gather_ps((float*)data.data(), offsets, 1);
    for (int i = 0; i < 8; i++) {
      std::cout << f000[i] << " ";
    }
    std::cout << "\n";
  }
}

TEST_CASE("truncate_to_int", "[vectorclass]") {
  float x = 0.65;
  Vec8f v(x);
  auto vn = truncate_to_int(v + 0.5);

  Logger::print_info("{}", vn[0]);
}

TEST_CASE("testing simd_buffer", "[simd]") {
  int N1 = 10, N2 = 8, N3 = 20;
  Grid g(N1, N2, N3);
  multi_array<simd::simd_buffer> array(g.extent());

  Logger::print_info("simd_buffer multi_array has pitch {}",
                     array.pitch());
  Logger::print_info("simd_buffer has size {}",
                     sizeof(simd::simd_buffer));

  array.assign(simd::simd_buffer{1.0f});

  REQUIRE(array(5, 3, 3).x[3] == 1.0f);
}

#endif
