#ifndef _FIELDS_H_
#define _FIELDS_H_

#include "constant_defs.h"
#include "data/enum_types.h"
#include "data/grid.h"
#include "data/multi_array.h"
#include "data/stagger.h"
#include "data/typedefs.h"
#include <array>
// #include "initial_conditions/initial_condition.h"

namespace Aperture {

/// This is the base class for fields living on a grid. It maintains a
/// \ref grid object and caches its linear size. It also implements a
/// function to check whether two grid extents are the same.
class FieldBase {
 public:
  /// Default constructor, initialize an empty grid and zero grid size.
  FieldBase() : m_grid(nullptr), m_grid_size(0) {}

  /// Main constructor, initializes the grid according to a given \ref
  /// grid object.
  FieldBase(const Grid &grid)
      : m_grid(&grid), m_grid_size(grid.size()) {}

  /// Destructor. No need to destruct anything since we didn't
  /// allocate dynamic memory. However it is useful to declare this as
  /// virtual since we need to derive from this class.
  virtual ~FieldBase() {}

  // Accessor methods for everything
  const Grid &grid() const { return *m_grid; }
  int grid_size() const { return m_grid_size; }
  Extent extent() const { return m_grid->extent(); }

 protected:
  const Grid *m_grid;   ///< Grid that this field is defined on
  int m_grid_size = 0;  //!< Cache the grid size for easier retrieval

  void check_grid_extent(const Extent &ext1, const Extent &ext2) const;
};  // ----- end of class field_base -----

/// Class for a scalar field with one component.
template <typename T>
class ScalarField : public FieldBase {
 public:
  typedef T data_type;
  typedef Grid grid_type;
  typedef MultiArray<T> array_type;
  typedef ScalarField<T> self_type;

  // Constructors and destructor
  ScalarField();
  ScalarField(const grid_type &grid, Stagger stagger = Stagger(0));
  ScalarField(const self_type &field);
  ScalarField(self_type &&field);
  virtual ~ScalarField();

  // Core functions
  void initialize();
  template <typename Func>
  void initialize(const Func &f);

  void assign(data_type value);
  void copyFrom(const self_type &field);
  self_type &operator=(const self_type &field);
  self_type &operator=(self_type &&field);

  void resize(const grid_type &grid);

  // Arithmetic operations
  self_type &multiplyBy(data_type value);
  self_type &multiplyBy(const ScalarField<T> &field);
  self_type &addBy(data_type value);
  self_type &addBy(const ScalarField<T> &field);
  self_type &subtractBy(data_type value);
  self_type &subtractBy(const ScalarField<T> &field);

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
  data_type &operator()(const Index &idx) {
    return m_array(idx.x, idx.y, idx.z);
  }
  const data_type &operator()(const Index &idx) const {
    return m_array(idx.x, idx.y, idx.z);
  }

  // Accessor methods
  array_type &data() { return m_array; }
  const array_type &data() const { return m_array; }
  cudaPitchedPtr &ptr() { return m_array.data_d(); }
  cudaPitchedPtr ptr() const { return m_array.data_d(); }
  Stagger stagger() const { return m_stagger; }

  void set_stagger(Stagger stagger) { m_stagger = stagger; }
  void sync_to_device() { m_array.sync_to_device(); }
  void sync_to_host() { m_array.sync_to_host(); }

 private:
  array_type m_array;
  Stagger m_stagger;
};  // ----- end of class scalar_field -----

template <typename T>
class VectorField : public FieldBase {
 public:
  typedef T data_type;
  typedef Grid grid_type;
  typedef MultiArray<T> array_type;
  typedef VectorField<T> self_type;

  /// Constructors and Destructor
  VectorField();
  VectorField(const grid_type &grid);
  VectorField(const self_type &field);
  VectorField(self_type &&field);
  virtual ~VectorField();

  self_type &operator=(const self_type &field);
  self_type &operator=(self_type &&field);

  /// Core functions
  void initialize();
  template <typename Func>
  void initialize(int component, const Func &f);
  template <typename Func>
  void initialize(const Func &f);
  // void initialize(const initial_condition &ic, FieldType type);

  void assign(data_type value, int n);
  void assign(data_type value);
  void assign(const VectorField<T>& field, T q);
  void copyFrom(const self_type &field);

  void resize(const grid_type &grid);
  // void init_array_ptrs();

  /// Arithmetic operations
  self_type &multiplyBy(data_type value);
  self_type &multiplyBy(const ScalarField<T> &field);
  self_type &addBy(data_type value, int n);
  self_type &addBy(const VectorField<T> &field);
  self_type &addBy(const VectorField<T> &field, T q);
  self_type &subtractBy(data_type value, int n);
  self_type &subtractBy(const VectorField<T> &field);

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
    return m_array[n](idx.x, idx.y, idx.z);
  }
  const data_type &operator()(int n, const Index &idx) const {
    return m_array[n](idx.x, idx.y, idx.z);
  }

  /// Setting ptr. Pointer is unmanaged!
  // void set_ptr(int n, T* p) { m_array[n].set_data(p); }

  /// Accessor methods
  array_type &data(int n) { return m_array[n]; }
  const array_type &data(int n) const { return m_array[n]; }
  cudaPitchedPtr ptr(int n) { return m_array[n].data_d(); }
  const cudaPitchedPtr ptr(int n) const { return m_array[n].data_d(); }
  // data_type **array_ptrs() { return m_ptrs; }
  // const data_type *const *array_ptrs() const { return m_ptrs; }
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
  void sync_to_device() {
    for (int i = 0; i < VECTOR_DIM; i++) m_array[i].sync_to_device();
  }
  void sync_to_host() {
    for (int i = 0; i < VECTOR_DIM; i++) m_array[i].sync_to_host();
  }

 private:
  std::array<array_type, VECTOR_DIM> m_array;
  std::array<Stagger, VECTOR_DIM> m_stagger;
  FieldType m_type;
  // Keep a cached array of pointers to the actual device pointers,
  // convenient for the kernels
  // data_type **m_ptrs;

  // Default normalization is coord FieldNormalization
  // m_normalization = FieldNormalization::coord;
};  // ----- end of class vector_field -----

}  // namespace Aperture

#include "data/detail/fields_impl.hpp"

#endif  // _FIELDS_H_
