#ifndef _FIELDS_H_
#define _FIELDS_H_

#include "core/constant_defs.h"
#include "core/enum_types.h"
#include "core/grid.h"
#include "core/multi_array.h"
#include "core/stagger.h"
#include "core/typedefs.h"
#include <array>

namespace Aperture {

template <int N, typename T>
class field {
 public:
  typedef field<N, T> self_type;
  typedef multi_array<T> array_type;

  field();
  field(const Grid& grid);
  field(const self_type& other);
  field(self_type&& other);
  virtual ~field();

  self_type &operator=(const self_type &field);
  self_type &operator=(self_type &&field);

  void initialize();
  template <typename Func>
  void initialize(int component, const Func &f);
  template <typename Func>
  void initialize(const Func &f);

  void assign(T value, int n);
  void assign(T value);
  void copy_from(const self_type &field);
  void resize(const Grid &grid);

  void set_field_type(FieldType type);

  T &operator()(int x, int y = 0, int z = 0) {
    return m_array[0](x, y, z);
  }
  const T &operator()(int x, int y = 0,
                              int z = 0) const {
    return m_array[0](x, y, z);
  }
  T &operator()(int n, int x, int y = 0, int z = 0) {
    return m_array[n](x, y, z);
  }
  const T &operator()(int n, int x, int y = 0,
                              int z = 0) const {
    return m_array[n](x, y, z);
  }
  T &operator()(const Index &idx) {
    return m_array[0](idx.x, idx.y, idx.z);
  }
  const T &operator()(const Index &idx) const {
    return m_array[0](idx.x, idx.y, idx.z);
  }
  T &operator()(int n, const Index &idx) {
    return m_array[n](idx.x, idx.y, idx.z);
  }
  const T &operator()(int n, const Index &idx) const {
    return m_array[n](idx.x, idx.y, idx.z);
  }

  const Grid& grid() const { return *m_grid; }
  Extent extent() const { return m_grid->extent(); }
  size_t grid_size() const { return m_grid->extent().size(); }

  array_type &data(int n = 0) { return m_array[n]; }
  const array_type &data(int n = 0) const { return m_array[n]; }

  Stagger stagger(int n = 0) const { return m_stagger[n]; }

  void set_stagger(int n, Stagger stagger) { m_stagger[n] = stagger; }

  void copy_to_device() {
    for (int i = 0; i < N; i++) m_array[i].copy_to_device();
  }
  void copy_to_device(int n) {
    m_array[n].copy_to_device();
  }
  void copy_to_host() {
    for (int i = 0; i < N; i++) m_array[i].copy_to_host();
  }
  void copy_to_host(int n) {
    m_array[n].copy_to_host();
  }

 protected:
  const Grid* m_grid;
  std::array<multi_array<T>, N> m_array;
  std::array<Stagger, N> m_stagger;

  void check_grid_extent(const Extent& ext1, const Extent& ext2) const {
    if (ext1 != ext2)
      throw std::invalid_argument("Field grids don't match!");
  }
  void check_component_range(int n) const;
};

template <typename T>
using scalar_field = field<1, T>;

template <typename T>
using vector_field = field<3, T>;

template <int N, typename T>
template <typename Func>
void
field<N, T>::initialize(int component, const Func& f) {
  check_component_range(component);
  auto& mesh = m_grid->mesh();
  // This way vector field is always defined in the center of the cell
  // face, staggered in the direction of the component
  for (int k = 0; k < m_grid->extent().depth(); ++k) {
    double x3 = mesh.pos(2, k, m_stagger[component][2]);
    size_t k_offset =
        k * mesh.dims[0] * mesh.dims[1];
    for (int j = 0; j < m_grid->extent().height(); ++j) {
      double x2 = mesh.pos(1, j, m_stagger[component][1]);
      size_t j_offset = j * mesh.dims[0];
#pragma omp simd
      for (int i = 0; i < m_grid->extent().width(); ++i) {
        double x1 = mesh.pos(0, i, m_stagger[component][0]);
        m_array[component][i + j_offset + k_offset] = f(x1, x2, x3);
      }
    }
  }
  copy_to_device(component);
}

template <int N, typename T>
template <typename Func>
void
field<N, T>::initialize(const Func& f) {
  for (int i = 0; i < N; i++) {
    initialize(i, [&f, i](T x1, T x2, T x3) { return f(i, x1, x2, x3); });
  }
}

}  // namespace Aperture

#endif  // _FIELDS_H_
