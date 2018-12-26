#ifndef _FIELDS_IMPL_H_
#define _FIELDS_IMPL_H_

#include "data/fields.h"

namespace Aperture {

template <typename T>
template <typename Func>
void
scalar_field<T>::initialize(const Func& f) {
  // This way scalar field is always defined in the center of the cell
  for (int k = 0; k < m_grid->extent().depth(); ++k) {
    double x3 = m_grid->mesh().pos(2, k, m_stagger[2]);
    size_t k_offset = k * m_array.pitch() * m_array.height();
    for (int j = 0; j < m_grid->extent().height(); ++j) {
      double x2 = m_grid->mesh().pos(1, j, m_stagger[1]);
      size_t j_offset = j * m_array.pitch();
#pragma omp simd
      for (int i = 0; i < m_grid->extent().width(); ++i) {
        double x1 = m_grid->mesh().pos(0, i, m_stagger[0]);
        m_array[i * sizeof(T) + j_offset + k_offset] = f(x1, x2, x3);
      }
    }
  }
}

template <typename T>
template <typename Func>
void
vector_field<T>::initialize(int component, const Func& f) {
  // This way vector field is always defined in the center of the cell
  // face, staggered in the direction of the component
  for (int k = 0; k < m_grid->extent().depth(); ++k) {
    double x3 = m_grid->mesh().pos(2, k, m_stagger[component][2]);
    size_t k_offset = k * m_array[component].pitch() * m_array[component].height();
    for (int j = 0; j < m_grid->extent().height(); ++j) {
      double x2 = m_grid->mesh().pos(1, j, m_stagger[component][1]);
      size_t j_offset = j * m_array[component].pitch();
#pragma omp simd
      for (int i = 0; i < m_grid->extent().width(); ++i) {
        double x1 = m_grid->mesh().pos(0, i, m_stagger[component][0]);
        m_array[component][i * sizeof(T) + j_offset + k_offset] =
            f(x1, x2, x3);
      }
    }
  }
}

template <typename T>
template <typename Func>
void
vector_field<T>::initialize(const Func& f) {
  initialize(0, [&f](T x1, T x2, T x3) { return f(0, x1, x2, x3); });
  initialize(1, [&f](T x1, T x2, T x3) { return f(1, x1, x2, x3); });
  initialize(2, [&f](T x1, T x2, T x3) { return f(2, x1, x2, x3); });
}

}  // namespace Aperture

#endif  // _FIELDS_IMPL_H_
