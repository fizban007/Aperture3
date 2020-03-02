#ifndef __FIELDS_IMPL_H_
#define __FIELDS_IMPL_H_

#include "fields.h"

namespace Aperture {

template <int N, typename T>
field<N, T>::field() : m_grid(nullptr) {}

template <int N, typename T>
field<N, T>::field(const Grid& grid) : m_grid(&grid) {
  for (int i = 0; i < N; i++) {
    m_array[i] = multi_array<T>(grid.extent());
    m_stagger[i] = Stagger(0b111);
  }
  initialize();
}

// template <int N, typename T>
// field<N, T>::field(const self_type& other) : m_grid(other.m_grid) {
//   for (int i = 0; i < N; i++) {
//     m_array[i] = other.m_array[i];
//     m_stagger[i] = other.m_stagger[i];
//   }
// }

template <int N, typename T>
field<N, T>::field(self_type&& other) : m_grid(other.m_grid) {
  for (int i = 0; i < N; i++) {
    m_array[i] = std::move(other.m_array[i]);
    m_stagger[i] = other.m_stagger[i];
  }
}

template <int N, typename T>
field<N, T>::~field() {}

// template <int N, typename T>
// field<N, T>&
// field<N, T>::operator=(const self_type& other) {
//   m_grid = other.m_grid;

//   for (int i = 0; i < N; i++) {
//     m_array[i] = other.m_array[i];
//     m_stagger[i] = other.m_stagger[i];
//   }
//   return *this;
// }

template <int N, typename T>
field<N, T>&
field<N, T>::operator=(self_type&& other) {
  m_grid = other.m_grid;

  for (int i = 0; i < N; i++) {
    m_array[i] = std::move(other.m_array[i]);
    m_stagger[i] = other.m_stagger[i];
  }
  return *this;
}

template <int N, typename T>
void
field<N, T>::resize(const Grid &grid) {
  m_grid = &grid;

  for (int i = 0; i < N; i++) {
    m_array[i].resize(grid.extent());
  }
}

template <int N, typename T>
void
field<N, T>::copy_from(const self_type &field) {
  // We can copy as long as the extents are the same
  check_grid_extent(m_grid->extent(), field.grid().extent());
  for (int i = 0; i < N; i++) {
    m_array[i].copy_from(field.m_array[i]);
  }
}

template <int N, typename T>
void
field<N, T>::set_field_type(FieldType type) {
  if (N == 3) {
    if (type == FieldType::E) {
      set_stagger(0, Stagger(0b110));
      set_stagger(1, Stagger(0b101));
      set_stagger(2, Stagger(0b011));
    } else if (type == FieldType::B) {
      set_stagger(0, Stagger(0b001));
      set_stagger(1, Stagger(0b010));
      set_stagger(2, Stagger(0b100));
    }
  }
}

template <int N, typename T>
void
field<N, T>::check_component_range(int n) const {
  if (n < 0 || n >= N) {
    throw std::out_of_range(
        "Trying to access a non-existent field component!");
  }
}

}  // namespace Aperture

#endif // __FIELDS_IMPL_H_
