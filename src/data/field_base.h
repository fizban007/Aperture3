#ifndef _FIELD_BASE_H_
#define _FIELD_BASE_H_

#include "core/grid.h"

namespace Aperture {

/// This is the base class for fields living on a grid. It maintains a
/// \ref grid object and caches its linear size. It also implements a
/// function to check whether two grid extents are the same.
class field_base {
 public:
  /// Default constructor, initialize an empty grid and zero grid size.
  field_base() : m_grid(nullptr), m_grid_size(0) {}

  /// Main constructor, initializes the grid according to a given \ref
  /// grid object.
  field_base(const Grid &grid)
      : m_grid(&grid), m_grid_size(grid.size()) {}

  /// Destructor. No need to destruct anything since we didn't
  /// allocate dynamic memory. However it is useful to declare this as
  /// virtual since we need to derive from this class.
  virtual ~field_base() {}

  // Accessor methods for everything
  const Grid &grid() const { return *m_grid; }
  int grid_size() const { return m_grid_size; }
  Extent extent() const { return m_grid->extent(); }

 protected:
  const Grid *m_grid;   ///< Grid that this field is defined on
  int m_grid_size = 0;  //!< Cache the grid size for easier retrieval

  void check_grid_extent(const Extent &ext1, const Extent &ext2) const {
    if (ext1 != ext2)
      throw std::invalid_argument("Field grids don't match!");
  }
};  // ----- end of class field_base -----

} // namespace Aperture

#endif  // _FIELD_BASE_H_
