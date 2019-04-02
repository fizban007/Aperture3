#ifndef _FIELDS_DEV_IMPL_H_
#define _FIELDS_DEV_IMPL_H_

#include "cuda/data/fields_dev.h"
#include "utils/logger.h"

namespace Aperture {

template <typename T>
template <typename Func>
void
cu_scalar_field<T>::initialize(const Func& f) {
  // This way scalar field is always defined in the center of the cell
  for (int k = 0; k < m_grid->extent().depth(); ++k) {
    double x3 = m_grid->mesh().pos(2, k, m_stagger[2]);
    for (int j = 0; j < m_grid->extent().height(); ++j) {
      double x2 = m_grid->mesh().pos(1, j, m_stagger[1]);
      for (int i = 0; i < m_grid->extent().width(); ++i) {
        double x1 = m_grid->mesh().pos(0, i, m_stagger[0]);
        m_array(i, j, k) = f(x1, x2, x3);
      }
    }
  }
  m_array.sync_to_device();
}

template <typename T>
template <typename Func>
void
cu_vector_field<T>::initialize(int component, const Func& f) {
  // This way vector field is always defined in the center of the cell
  // face, staggered in the direction of the component
  for (int k = 0; k < m_grid->extent().depth(); ++k) {
    double x3 = m_grid->mesh().pos(2, k, m_stagger[component][2]);
    for (int j = 0; j < m_grid->extent().height(); ++j) {
      double x2 = m_grid->mesh().pos(1, j, m_stagger[component][1]);
      for (int i = 0; i < m_grid->extent().width(); ++i) {
        double x1 = m_grid->mesh().pos(0, i, m_stagger[component][0]);
        m_array[component](i, j, k) = f(x1, x2, x3);
        // Logger::print_debug("x is ({}, {}, {}), f is {}", x1, x2, x3, f(x1, x2, x3));
      }
    }
  }
  m_array[component].sync_to_device();
}

template <typename T>
template <typename Func>
void
cu_vector_field<T>::initialize(const Func& f) {
  initialize(0, [&f](T x1, T x2, T x3) { return f(0, x1, x2, x3); });
  initialize(1, [&f](T x1, T x2, T x3) { return f(1, x1, x2, x3); });
  initialize(2, [&f](T x1, T x2, T x3) { return f(2, x1, x2, x3); });
}

}  // namespace Aperture

#endif  // _FIELDS_DEV_IMPL_H_
