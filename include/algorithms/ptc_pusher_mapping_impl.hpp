#ifndef _PTC_PUSHER_MAPPING_IMPL_H_
#define _PTC_PUSHER_MAPPING_IMPL_H_

#include <immintrin.h>
#include <emmintrin.h>
#include "algorithms/ptc_pusher_mapping.h"
#include "utils/Logger.h"

namespace Aperture {

template <typename Metric>
void
ParticlePusher_Mapping::update_position_f::operator()(const Metric& metric, Particles& particles, const Grid& grid, double dt, Index_t start, Index_t num) const {
  auto& ptc = particles.data();
  // if (num == 0) num = particles.number() - start;
  for (Index_t idx = start; idx < start + num; idx++) {
    if (particles.is_empty(idx)) continue;

    Vec3<Mom_t> v(ptc.p1[idx], ptc.p2[idx], ptc.p3[idx]);
    v = v / ptc.gamma[idx];
    int cell = ptc.cell[idx];
    Vec3<float> x(ptc.x1[idx], ptc.x2[idx], ptc.x3[idx]);
    auto x_old = grid.mesh().pos_particle(cell, x);
    auto x_new = x_old;
    metric.VectorToCartesian(v, x_old);
    metric.PosToCartesian(x_new);

    x_new += v * dt;

    // TODO: Check if the position makes sense?
    metric.PosFromCartesian(x_new);

    ptc.dx1[idx] = x_new.x - x_old.x;
    ptc.dx2[idx] = x_new.y - x_old.y;
    ptc.dx3[idx] = x_new.z - x_old.z;

    metric.VectorFromCartesian(v, x_new);
    ptc.p1[idx] = v.x * ptc.gamma[idx];
    ptc.p2[idx] = v.y * ptc.gamma[idx];
    ptc.p3[idx] = v.z * ptc.gamma[idx];

    ptc.x1[idx] += ptc.dx1[idx];
    ptc.x2[idx] += ptc.dx2[idx];

    int delta_cell = (int)std::floor(ptc.x1[idx]);
    cell += delta_cell * grid.mesh().idx_increment(0);
    ptc.x1[idx] -= (float)delta_cell;

    delta_cell = (int)std::floor(ptc.x2[idx]);
    cell += delta_cell * grid.mesh().idx_increment(1);
    ptc.x2[idx] -= (float)delta_cell;

    if (grid.dim() > 2) {
      ptc.x3[idx] += ptc.dx3[idx];

      delta_cell = (int)std::floor(ptc.x3[idx]);
      cell += delta_cell * grid.mesh().idx_increment(2);
      ptc.x3[idx] -= (float)delta_cell;
    }
    ptc.cell[idx] = cell;
    // for (unsigned int i = 0; i < 3; i++) {
    //   ptc.x[idx][i] = x_new[i];
    //   if (i < grid.dim()) {
    //     int delta_cell = (int)std::floor(ptc.x[idx][i]);
    //     cell += delta_cell * grid.mesh().idx_increment(i);
    //     ptc.x[idx][i] -= (Pos_t)delta_cell;
    //     ptc.cell[idx] = cell;

    //     if (delta_cell > 1 || delta_cell < -1) {
    //       Logger::err( "UpdatePosition Error: Cell crossing! Dir = ", i,
    //                    ", cell = ", cell, ", x = ", ptc.x[idx].vec3(), "dx = ", ptc.dx[idx].vec3() );
    //     }
    //   }
    // }
  }
}

template <typename Metric>
void
ParticlePusher_Mapping::update_position_avx_f::operator()(const Metric& metric, ParticlePusher_Mapping& pusher, Particles& particles, const Grid& grid, double dt) const {
#if defined(__AVX2__) && (defined(__ICC) || defined(__INTEL_COMPILER))
    auto& ptc = particles.data();
    Index_t idx;
    for (idx = 0; idx + 3 < particles.number(); idx += 4) {
      __m256d gamma = _mm256_load_pd(ptc.gamma + idx);
      __m256d v1 = _mm256_load_pd(ptc.p1 + idx);
      __m256d v2 = _mm256_load_pd(ptc.p2 + idx);
      __m256d v3 = _mm256_load_pd(ptc.p3 + idx);

      v1 = _mm256_div_pd(v1, gamma);
      v2 = _mm256_div_pd(v2, gamma);
      v3 = _mm256_div_pd(v3, gamma);

      __m128 x1 = _mm_load_ps(ptc.x1 + idx);
      __m128 x2 = _mm_load_ps(ptc.x2 + idx);
      __m128 x3 = _mm_load_ps(ptc.x3 + idx);

      __m128i cell = _mm_load_si128((const __m128i*)(ptc.cell + idx));
      // __m128i c1 = grid.mesh().get_c1(cell);
      // __m128i c2 = grid.mesh().get_c2(cell);
      // __m128i c3 = grid.mesh().get_c3(cell);

      __m256d x_sph_1 = grid.mesh().pos(0, grid.mesh().get_c1(cell), x1);
      __m256d x_sph_2 = grid.mesh().pos(0, grid.mesh().get_c2(cell), x2);
      __m256d x_sph_3 = grid.mesh().pos(0, grid.mesh().get_c3(cell), x3);
      __m256d x_old_1 = x_sph_1;
      __m256d x_old_2 = x_sph_2;
      __m256d x_old_3 = x_sph_3;

      metric.VectorToCartesian(&v1, &v2, &v3, &x_sph_1, &x_sph_2, &x_sph_3);
      metric.PosToCartesian(&x_sph_1, &x_sph_2, &x_sph_3);

      x_sph_1 = _mm256_fmadd_pd(v1, _mm256_set1_pd(dt), x_sph_1);
      x_sph_2 = _mm256_fmadd_pd(v2, _mm256_set1_pd(dt), x_sph_2);
      x_sph_3 = _mm256_fmadd_pd(v3, _mm256_set1_pd(dt), x_sph_3);

      metric.PosFromCartesian(&x_sph_1, &x_sph_2, &x_sph_3);
      metric.VectorFromCartesian(&v1, &v2, &v3, &x_sph_1, &x_sph_2, &x_sph_3);

      _mm256_store_pd(ptc.p1 + idx, _mm256_mul_pd(v1, gamma));
      _mm256_store_pd(ptc.p2 + idx, _mm256_mul_pd(v2, gamma));
      _mm256_store_pd(ptc.p3 + idx, _mm256_mul_pd(v3, gamma));

      __m128 dx1 = _mm256_cvtpd_ps(_mm256_sub_pd(x_sph_1, x_old_1));
      __m128 dx2 = _mm256_cvtpd_ps(_mm256_sub_pd(x_sph_2, x_old_2));

      x1 = _mm_add_ps(x1, dx1);
      x2 = _mm_add_ps(x2, dx2);

      _mm_store_ps(ptc.dx1 + idx, dx1);
      _mm_store_ps(ptc.dx2 + idx, dx2);

      if (grid.dim() > 2) {
        __m128 dx3 = _mm256_cvtpd_ps(_mm256_sub_pd(x_sph_3, x_old_3));
        x3 = _mm_add_ps(x3, dx3);
        _mm_store_ps(ptc.dx3 + idx, dx3);
      }

      __m128 delta_cell = _mm_floor_ps(x1);
      x1 = _mm_sub_ps(x1, delta_cell);
      cell = _mm_add_epi32(cell, _mm_mul_epi32(_mm_cvtps_epi32(delta_cell), _mm_set1_epi32(grid.mesh().idx_increment(0))));

      delta_cell = _mm_floor_ps(x2);
      x2 = _mm_sub_ps(x2, delta_cell);
      cell = _mm_add_epi32(cell, _mm_mul_epi32(_mm_cvtps_epi32(delta_cell), _mm_set1_epi32(grid.mesh().idx_increment(1))));

      if (grid.dim() > 2) {
        delta_cell = _mm_floor_ps(x3);
        x3 = _mm_sub_ps(x3, delta_cell);
        cell = _mm_add_epi32(cell, _mm_mul_epi32(_mm_cvtps_epi32(delta_cell), _mm_set1_epi32(grid.mesh().idx_increment(2))));

        _mm_store_ps(ptc.x3 + idx, x3);
      }

      _mm_store_ps(ptc.x1 + idx, x1);
      _mm_store_ps(ptc.x2 + idx, x2);
      _mm_store_si128((__m128i*)(ptc.cell + idx), cell);
    }
    pusher.update_position(metric, particles, grid, dt, idx, particles.number() - idx);
#endif
}

}

#endif  // _PTC_PUSHER_MAPPING_IMPL_H_
