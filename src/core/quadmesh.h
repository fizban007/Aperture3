#ifndef _QUADMESH_H_
#define _QUADMESH_H_

#include "core/constant_defs.h"
#include "cuda/cuda_control.h"
#include "core/stagger.h"
#include "core/typedefs.h"
#include "core/vec3.h"
// #include "utils/simd.h"
#include <algorithm>
// #include <immintrin.h>
#include <iomanip>

namespace Aperture {

struct Quadmesh {
  int dims[3];   //!< Dimensions of the grid of each direction
  int guard[3];  //!< Number of guard cells at either end of each
                 //!< direction

  Scalar delta[3];  //!< Grid spacing on each direction (spacing in
                    //!< coordinate space)
  Scalar inv_delta[3];
  Scalar lower[3];  //!< Lower limit of the grid on each direction
  Scalar sizes[3];  //!< Size of the grid in coordinate space

  int tileSize[3];
  int offset[3];
  int dimension;

  HOST_DEVICE Quadmesh() {  //!< Default constructor
// Only define an empty constructor when compiling with Cuda enabled.
// This allows declaring a quadmesh in __constant__ memory.
#ifndef __CUDACC__
    // Initialize all quantities to zero, and dimensions to 1
    for (int i = 0; i < 3; i++) {
      dims[i] = 1;
      guard[i] = 0;
      delta[i] = 1.0;
      lower[i] = 0.0;
      sizes[i] = 0.0;
      tileSize[i] = 1;
      inv_delta[i] = 1.0;
      offset[i] = 0;
    }
    dimension = 1;
#endif  // __CUDACC__
  }

  ///  Constructor which only initialize dimensions.
  HOST_DEVICE Quadmesh(int N1, int N2 = 1, int N3 = 1) {
    dims[0] = (N1 > 1 ? N1 : 1);
    dims[1] = (N2 > 1 ? N2 : 1);
    dims[2] = (N3 > 1 ? N3 : 1);

    // Initialize other quantities to zero
    for (int i = 0; i < 3; i++) {
      guard[i] = 0;
      delta[i] = 1.0;
      inv_delta[i] = 1.0;
      lower[i] = 0.0;
      sizes[i] = delta[i] * dims[i];
      tileSize[i] = 1;
      offset[i] = 0;
    }
    dimension = dim();
  }

  ///  Assignment operator
  HOST_DEVICE Quadmesh& operator=(const Quadmesh& m) {
    for (int i = 0; i < 3; i++) {
      dims[i] = m.dims[i];
      guard[i] = m.guard[i];
      delta[i] = m.delta[i];
      inv_delta[i] = m.inv_delta[i];
      lower[i] = m.lower[i];
      sizes[i] = m.sizes[i];
      tileSize[i] = m.tileSize[i];
      offset[i] = m.offset[i];
    }
    dimension = m.dimension;
    return *this;
  }

  ///  Comparison operator
  HOST_DEVICE bool operator==(const Quadmesh& m) const {
    bool result = true;
    for (int i = 0; i < 3; i++) {
      result = result && (dims[i] == m.dims[i]);
      result = result && (guard[i] == m.guard[i]);
      result = result && (sizes[i] == m.sizes[i]);
      result = result && (lower[i] == m.lower[i]);
      result = result && (offset[i] == m.offset[i]);
      // result = result && (tileSize[i] == m.tileSize[i]);
    }
    result = result && (dimension == m.dimension);
    return result;
  }

  ///  Reduced dimension in one direction.
  ///
  ///  Reduced dimension means the total size of the grid minus the
  ///  guard cells in both ends. This function is only defined for i >=
  ///  0 and i < DIM.
  HD_INLINE int reduced_dim(int i) const {
    return (dims[i] - 2 * guard[i]);
  }

  ///  Coordinate of a point inside cell n in dimension i.
  ///
  ///  This function applies to field points. Stagger = false means
  ///  field is defined at cell center, while stagger = true means field
  ///  defined at cell boundary at the end.
  ///
  ///  This calculation is assuming boundary located at the interface
  ///  between guard cells and physical cells. The function is only
  ///  defined for i >= 0 and i < DIM.
  HD_INLINE Scalar pos(int i, int n, bool stagger) const {
    return pos(i, n, (int)stagger);
  }

  ///  Coordinate of a point inside cell n in dimension i.
  ///
  ///  This function applies to field points. Stagger = 0 means field is
  ///  defined at cell center, while stagger = 1 means field defined at
  ///  cell boundary at the end.
  ///
  ///  This calculation is assuming boundary located at the interface
  ///  between guard cells and physical cells. The function is only
  ///  defined for i >= 0 and i < DIM.
  HD_INLINE Scalar pos(int i, int n, int stagger) const {
    return pos(i, n, (Scalar)(stagger * 0.5 + 0.5));
  }

  ///  Coordinate of a point inside cell n in dimension i.
  ///
  ///  This function applies to particles. pos_in_cell is the relative
  ///  position of the particle in the cell and varies from 0.0 to 1.0.
  ///
  ///  This calculation is assuming boundary located at the interface
  ///  between guard cells and physical cells. The function is only
  ///  defined for i >= 0 and i < DIM.
  HD_INLINE Scalar pos(int i, int n, Scalar pos_in_cell) const {
    if (i < dimension)
      return (lower[i] + delta[i] * (n - guard[i] + pos_in_cell));
    else
      // return 0.0;
      return pos_in_cell * delta[i];
  }

  HD_INLINE Vec3<Scalar> pos_3d(int idx, Stagger stagger) const {
    Vec3<Scalar> result;
    result[0] = pos(0, idx % dims[0], stagger[0]);
    result[1] = pos(1, (idx / dims[0]) % dims[1], stagger[1]);
    result[2] = pos(2, idx / (dims[0] * dims[1]), stagger[2]);
    return result;
  }

  ///  Full coordinate of a point inside the grid.
  ///
  ///  This function applies for only particle position. Pos_rel has
  ///  components greater than zero and less than the corresponding
  ///  delta.
  ///
  ///  This calculation is assuming boundary located at the interface
  ///  between guard cells and physical cells. The function is only
  ///  defined for i >= 0 and i < DIM.
  template <typename T>
  HD_INLINE Vec3<Scalar> pos_particle(int cell_linear,
                                      const Vec3<T>& pos_rel) const {
    Vec3<Scalar> pos_full(pos(0, get_c1(cell_linear), pos_rel.x),
                          pos(1, get_c2(cell_linear), pos_rel.y),
                          pos(2, get_c3(cell_linear),
                              pos_rel.z));  // Note deltas cannot be
                                            // zero, even that dimension
                                            // is absent.
    return pos_full;
  }

  ///  Upper boundary position in direction i
  HD_INLINE Scalar upper(int i) const {
    return pos(i, dims[i] - guard[i] - 1, 1);
  }

  ///  Find the relative position and cell number in the dual grid
  template <typename T>
  HD_INLINE void pos_dual(Vec3<int>& c, Vec3<T>& pos) const {
    for (int i = 0; i < dimension; i++) {
      if (pos[i] > 0.5) {
        pos[i] -= 0.5;
      } else {
        pos[i] += 0.5;
        c[i] -= 1;
      }
    }
  }

  ///  Index of the point if the grid is stratified into 1 direction.
  HD_INLINE int get_idx(int c1, int c2 = 0, int c3 = 0) const {
    return c1 + c2 * dims[0] + c3 * dims[0] * dims[1];
  }

  ///  Index of the point if the grid is stratified into 1 direction.
  HD_INLINE int get_idx(const Index& idx) const {
    return idx[0] + idx[1] * dims[0] + idx[2] * dims[0] * dims[1];
  }

  ///  Index increment in the particular direction
  HD_INLINE int idx_increment(int direction) const {
    if (direction >= dimension) return 0;
    switch (direction) {
      case 0:
        return 1;
      case 1:
        return dims[0];
      case 2:
        return dims[0] * dims[1];
      default:
        return 0;
    }
  }

  ///  Test if a point is inside the grid.
  HD_INLINE bool is_in_grid(int c1, int c2 = 0, int c3 = 0) const {
    return (c1 >= 0 && c1 < dims[0]) && (c2 >= 0 && c2 < dims[1]) &&
           (c3 >= 0 && c3 < dims[2]);
  }

  ///  Test if a point is inside the bulk of the grid, not in guard
  ///  cells.
  HD_INLINE bool is_in_bulk(int c1, int c2, int c3 = 0) const {
    return (c1 >= guard[0] && c1 < dims[0] - guard[0]) &&
           (c2 >= guard[1] && c2 < dims[1] - guard[1]) &&
           (c3 >= guard[2] && c3 < dims[2] - guard[2]);
  }

  ///  Test if a point is inside the bulk of the grid, not in guard
  ///  cells.
  HD_INLINE bool is_in_bulk(const Index& idx) const {
    // return (idx.x >= guard[0] && idx.x < dims[0] - guard[0])
    //     && (idx.y >= guard[1] && idx.y < dims[1] - guard[1])
    //     && (idx.z >= guard[2] && idx.z < dims[2] - guard[2]);
    return is_in_bulk(idx.x, idx.y, idx.z);
  }

  ///  Test if a point is inside the bulk of the grid, not in guard
  ///  cells.
  HD_INLINE bool is_in_bulk(int c) const {
    return is_in_bulk(get_c1(c), get_c2(c), get_c3(c));
  }

  ///  Get the size of the grid (product of all dimensions).
  HD_INLINE int size() const {
    int tmp = dims[0] * dims[1] * dims[2];
    return tmp;
  }

  ///  Find the zone the cell belongs to (for communication purposes)
  HD_INLINE int find_zone(int cell) const {
    int c1 = get_c1(cell);
    int c2 = get_c2(cell);
    int c3 = get_c3(cell);

    int z1 = (c1 >= guard[0]) + (c1 >= (dims[0] - guard[0]));
    int z2 = (c2 >= guard[1]) + (c2 >= (dims[1] - guard[1]));
    int z3 = (c3 >= guard[2]) + (c3 >= (dims[2] - guard[2]));
    return z1 + z2 * 3 + z3 * 9;
  }

  ///  Find the cell index from the global position, and get the
  ///  relative position as well.
  HD_INLINE int find_cell(const Vec3<Scalar>& pos,
                          Vec3<Pos_t>& rel_pos) const {
    int c1 = static_cast<int>(floor((pos.x - lower[0]) / delta[0])) +
             guard[0];
    // if (c1 < 0 || c1 > dims[0]) {
    //   std::cerr << "c1 out of range: " << c1 << std::endl;
    //   c1 = 0;
    // }
    int c2 = static_cast<int>(floor((pos.y - lower[1]) / delta[1])) +
             guard[1];
    if (dim() < 2) c2 = 0;
    // else if (c2 < 0 || c2 > dims[1]) {
    //   std::cerr << "c2 out of range: " << c2 << std::endl;
    //   c2 = 0;
    // }
    int c3 = static_cast<int>(floor((pos.z - lower[2]) / delta[2])) +
             guard[2];
    if (dim() < 3) c3 = 0;
    // else if (c3 < 0 || c3 > dims[2]) {
    //   std::cerr << "c3 out of range: " << c3 << std::endl;
    //   c3 = 0;
    // }
    rel_pos.x =
        (pos.x - (c1 - guard[0]) * delta[0] - lower[0]) / delta[0];
    rel_pos.y =
        (pos.y - (c2 - guard[1]) * delta[1] - lower[1]) / delta[1];
    rel_pos.z =
        (pos.z - (c3 - guard[2]) * delta[2] - lower[2]) / delta[2];
    return get_idx(c1, c2, c3);
  }

  ///  Get the extent of the grid. Used for interfacing with
  ///  multiarrays.
  HD_INLINE Extent extent() const {
    return Extent{dims[0], dims[1], dims[2]};
    //    return tmp;
  }

  HD_INLINE Extent extent_less() const {
    return Extent{dims[0] - 2 * guard[0], dims[1] - 2 * guard[1],
                  dims[2] - 2 * guard[2]};
  }

  HD_INLINE int num_tiles() const {
    int ret = reduced_dim(0) * reduced_dim(1) * reduced_dim(2);
    for (int i = 0; i < 3; i++)
      if (dims[i] > 1) ret /= tileSize[i];
    return ret;
  }

  HD_INLINE uint32_t get_c1(uint32_t idx) const {
    return idx % dims[0];
  }
  HD_INLINE uint32_t get_c2(uint32_t idx) const {
    return (idx / dims[0]) % dims[1];
  }
  HD_INLINE uint32_t get_c3(uint32_t idx) const {
    return idx / (dims[0] * dims[1]);
  }

  HD_INLINE Vec3<int> get_cell_3d(int idx) const {
    return Vec3<int>(get_c1(idx), get_c2(idx), get_c3(idx));
  }

  HD_INLINE uint32_t tile_id(int c1, int c2, int c3) const {
    uint32_t tileN1 = (dims[0] > 1 ? dims[0] / tileSize[0] : 1);
    uint32_t tileN2 = (dims[1] > 1 ? dims[1] / tileSize[1] : 1);
    int ret = (c1 - guard[0]) / tileSize[0];
    if (dims[1] > 1) ret += ((c2 - guard[1]) / tileSize[0]) * tileN1;
    if (dims[2] > 1)
      ret += ((c3 - guard[2]) / tileSize[0]) * tileN1 * tileN2;
    return ret;
  }

  HD_INLINE uint32_t tile_id(uint32_t cell) const {
    if (cell == MAX_CELL) return MAX_TILE;
    int c1 = get_c1(cell), c2 = get_c2(cell), c3 = get_c3(cell);
    return tile_id(c1, c2, c3);
  }

  HD_INLINE int dim() const {
    if (dims[1] <= 1 && dims[2] <= 1)
      return 1;
    else if (dims[2] <= 1)
      return 2;
    else
      return 3;
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const Quadmesh& mesh) {
    os << std::fixed
       << std::setprecision(std::numeric_limits<double>::digits10);
    os << mesh.dims[0] << " " << mesh.dims[1] << " " << mesh.dims[2]
       << " ";
    os << mesh.guard[0] << " " << mesh.guard[1] << " " << mesh.guard[2]
       << " ";
    os << mesh.delta[0] << " " << mesh.delta[1] << " " << mesh.delta[2]
       << " ";
    os << mesh.lower[0] << " " << mesh.lower[1] << " " << mesh.lower[2]
       << " ";
    os << mesh.sizes[0] << " " << mesh.sizes[1] << " " << mesh.sizes[2]
       << " ";
    // os << mesh.tileSize << " ";
    os << mesh.dimension << " ";

    return os;
  }

  friend std::istream& operator>>(std::istream& is, Quadmesh& mesh) {
    is >> mesh.dims[0] >> mesh.dims[1] >> mesh.dims[2];
    is >> mesh.guard[0] >> mesh.guard[1] >> mesh.guard[2];
    is >> mesh.delta[0] >> mesh.delta[1] >> mesh.delta[2];
    is >> mesh.lower[0] >> mesh.lower[1] >> mesh.lower[2];
    is >> mesh.sizes[0] >> mesh.sizes[1] >> mesh.sizes[2];
    // is >> mesh.tileSize;
    is >> mesh.dimension;

    char c;
    is.get(c);  // hack to get an additional position on fs
    return is;
  }
};
}  // namespace Aperture

#endif  // _QUADMESH_H_
