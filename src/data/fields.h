#ifndef _FIELDS_H_
#define _FIELDS_H_

#include "constant_defs.h"
#include "data/enum_types.h"
#include "data/field_base.h"
#include "data/grid.h"
#include "data/multi_array.h"
#include "data/stagger.h"
#include "data/typedefs.h"
// #include "vectorclass.h"
#include <array>
// #include "initial_conditions/initial_condition.h"

namespace Aperture {

/// Class for a scalar field with one component.
template <typename T>
class scalar_field : public field_base {
 public:
  typedef T data_type;
  typedef Grid grid_type;
  typedef multi_array<T> array_type;
  typedef scalar_field<T> self_type;

  // Constructors and destructor
  scalar_field();
  scalar_field(const grid_type &grid, Stagger stagger = Stagger(0));
  scalar_field(const self_type &field);
  scalar_field(self_type &&field);
  virtual ~scalar_field();

  // Core functions
  void initialize();
  template <typename Func>
  void initialize(const Func &f);

  void assign(data_type value);
  void copy_from(const self_type &field);
  self_type &operator=(const self_type &field);
  self_type &operator=(self_type &&field);

  void resize(const grid_type &grid);

  // Arithmetic operations
  self_type &multiplyBy(data_type value);
  self_type &multiplyBy_slow(data_type value);
  self_type &multiplyBy(const scalar_field<T> &field);
  self_type &multiplyBy_slow(const scalar_field<T> &field);
  self_type &addBy(data_type value);
  self_type &addBy(const scalar_field<T> &field);
  self_type &subtractBy(data_type value);
  self_type &subtractBy(const scalar_field<T> &field);
  self_type &divideBy(const scalar_field<T> &field);

  // template <int Order>
  T interpolate(const Vec3<int> &c, const Vec3<Pos_t> &rel_pos,
                int order = 1) const;

  // Index operator
  data_type &operator()(int x, int y = 0, int z = 0) {
    return m_array(x, y, z);
  }
  const data_type &operator()(int x, int y = 0, int z = 0) const {
    return m_array(x, y, z);
  }
  data_type &operator()(const Index &idx) { return m_array(idx); }
  const data_type &operator()(const Index &idx) const {
    return m_array(idx);
  }

  // Accessor methods
  array_type &data() { return m_array; }
  const array_type &data() const { return m_array; }
  Stagger stagger() const { return m_stagger; }

  void set_stagger(Stagger stagger) { m_stagger = stagger; }

 private:
  array_type m_array;
  Stagger m_stagger;
};  // ----- end of class scalar_field -----

template <typename T>
class vector_field : public field_base {
 public:
  typedef T data_type;
  typedef Grid grid_type;
  typedef multi_array<T> array_type;
  typedef vector_field<T> self_type;

  /// Constructors and Destructor
  vector_field();
  vector_field(const grid_type &grid);
  vector_field(const self_type &field);
  vector_field(self_type &&field);
  virtual ~vector_field();

  self_type &operator=(const self_type &field);
  self_type &operator=(self_type &&field);

  /// Core functions
  void initialize();
  template <typename Func>
  void initialize(int component, const Func &f);
  template <typename Func>
  void initialize(const Func &f);

  void assign(data_type value, int n);
  void assign(data_type value);
  void assign(const vector_field<T> &field, const T &q);
  void copy_from(const self_type &field);

  void resize(const grid_type &grid);
  // void init_array_ptrs();

  /// Arithmetic operations
  self_type &multiplyBy(data_type value);
  self_type &multiplyBy(const scalar_field<T> &field);
  self_type &addBy(data_type value, int n);
  self_type &addBy(const vector_field<T> &field);
  self_type &addBy(const vector_field<T> &field, T q);
  self_type &subtractBy(data_type value, int n);
  self_type &subtractBy(const vector_field<T> &field);

  // template <int Order>
  Vec3<T> interpolate(const Vec3<int> &c, const Vec3<Pos_t> &rel_pos,
                      int order = 1) const;

  // Interpolate the field to cell center and store the result to
  // @result
  void interpolate_to_center(self_type &result);

  // Interpolate the field from cell center to the stagger position
  // according to m_stagger, and store the result to @result
  void interpolate_from_center(self_type &result, Scalar q = 1.0);

  // Interpolate the field from cell center to the stagger position
  // according to m_stagger, and add the result to @result
  void interpolate_from_center_add(self_type &result, Scalar q = 1.0);

  // void recenter(self_type& output) const;

  self_type &convertToFlux();
  self_type &convertFromFlux();

  // void normalize(FieldNormalization normalization);

  /// Index operator
  data_type &operator()(int n, int x, int y = 0, int z = 0) {
    return m_array[n](x, y, z);
  }
  const data_type &operator()(int n, int x, int y = 0,
                              int z = 0) const {
    return m_array[n](x, y, z);
  }
  data_type &operator()(int n, const Index &idx) {
    return m_array[n](idx);
  }
  const data_type &operator()(int n, const Index &idx) const {
    return m_array[n](idx);
  }

  /// Accessor methods
  array_type &data(int n) { return m_array[n]; }
  const array_type &data(int n) const { return m_array[n]; }
  Stagger stagger(int n) const { return m_stagger[n]; }
  const std::array<Stagger, VECTOR_DIM> &stagger() const {
    return m_stagger;
  }
  std::array<Stagger, VECTOR_DIM> stagger_dual() const;

  void set_stagger(int n, Stagger stagger) { m_stagger[n] = stagger; }
  void set_stagger(const std::array<Stagger, VECTOR_DIM> &stagger) {
    m_stagger = stagger;
  }
  void set_field_type(FieldType type);

 private:
  std::array<array_type, VECTOR_DIM> m_array;
  std::array<Stagger, VECTOR_DIM> m_stagger;
  FieldType m_type;

  // Default normalization is coord FieldNormalization
  // m_normalization = FieldNormalization::coord;
};  // ----- end of class vector_field -----

}  // namespace Aperture

#include "data/detail/fields_impl.hpp"

#endif  // _FIELDS_H_
