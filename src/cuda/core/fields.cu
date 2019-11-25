#include "core/fields_impl.hpp"

namespace Aperture {

template <int N, typename T>
void
field<N, T>::assign(T value, int n) {
  check_component_range(n);
  m_array[n].assign_dev(value);
}

template <int N, typename T>
void
field<N, T>::assign(T value) {
  for (int i = 0; i < N; i++)
    m_array[i].assign_dev(value);
}

template <int N, typename T>
void
field<N, T>::initialize() {
  for (int i = 0; i < N; i++) {
    m_array[i].assign_dev(T{0});
  }
}

template class field<1, float>;
template class field<1, double>;
template class field<3, float>;
template class field<3, double>;

}
