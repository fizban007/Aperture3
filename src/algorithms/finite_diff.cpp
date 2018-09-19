#include "algorithms/finite_diff.h"
#include "data/vec3.h"
#include "utils/logger.h"
#include <iostream>

// Loading data a block at a time
const int block_size = 8;

using namespace Aperture;

constexpr int FiniteDiff::m_h[3];

inline void
multiply_mod(char mod_mult, Scalar& value, const Grid& grid, int idx,
             Stagger_t stagger) {
  if (check_bit(mod_mult, 0)) value *= grid.scales(0, stagger)[idx];
  if (check_bit(mod_mult, 1)) value *= grid.scales(1, stagger)[idx];
  if (check_bit(mod_mult, 2)) value *= grid.scales(2, stagger)[idx];
}

inline void
divide_mod(char mod_div, Scalar& value, const Grid& grid, int idx,
           Stagger_t stagger) {
  if (check_bit(mod_div, 0)) value /= grid.scales(0, stagger)[idx];
  if (check_bit(mod_div, 1)) value /= grid.scales(1, stagger)[idx];
  if (check_bit(mod_div, 2)) value /= grid.scales(2, stagger)[idx];
}

inline Scalar
diff_2nd(Scalar f[], Scalar delta) {
  return (f[1] - f[0]) / delta;
}

inline Scalar
diff_2nd_side(Scalar f[], Scalar delta, bool zero) {
  return (zero ? (3.0 * f[0] - f[1] / 3.0) / delta
               : (-2.0 * f[0] + 3.0 * f[1] - f[2]) / delta);
}

// inline Scalar diff_2nd_side (Scalar f[], Scalar delta) {
//   return (3.0 * f[0] - f[1] / 3.0) / delta;
// }

inline Scalar
diff_4th(Scalar f[], Scalar delta) {
  return ((f[2] - f[1]) * 9.0 / 8.0 - (f[3] - f[0]) / 24.0) / delta;
}

// this is the (1, rest)-type derivative
inline Scalar
diff_4th_side(Scalar f[], Scalar delta) {
  return (-f[0] * 11.0 / 12.0 + f[1] * 17.0 / 24.0 + f[2] * 3.0 / 8.0 -
          f[3] * 5.0 / 24.0 + f[4] / 24.0) /
         delta;
}

// this is the (0, all)-type derivative.
inline Scalar
diff_4th_side_mod(Scalar f[], Scalar delta, bool zero) {
  return (zero ? (f[0] * 35.0 / 8.0 - f[1] * 35.0 / 24.0 +
                  f[2] * 21.0 / 40.0 - f[3] * 5.0 / 56.0) /
                     delta
               : (-f[0] * 31.0 / 8.0 + f[1] * 229.0 / 24.0 -
                  f[2] * 75.0 / 8.0 + f[3] * 37.0 / 8.0 -
                  f[4] * 11.0 / 12.0) /
                     delta);
}

// FiniteDiff::FiniteDiff(const Grid& grid)
//     : m_grid(grid) {}
FiniteDiff::FiniteDiff() {}

FiniteDiff::~FiniteDiff() {}

void
FiniteDiff::derivative(const Grid& grid, const array_type& input,
                       array_type& output, unsigned int dir,
                       const Aperture::Index start,
                       const Aperture::Extent ext,
                       const Aperture::diff_params& params) {
  // Explanation of Derivative
  // 0. Derivative calculates the derivative of input and ADD it to
  // output.
  // 1. the passed in start and ext are the computational domain for an
  // input field unstaggered in all directions. The staggeredness is
  // taken care of in the Derivative automatically.
  // 2. the computational domain being defined as above, we always
  // requires that all fields that lie on the edges of this domain also
  // belong to the domain.
  // 3. range_start and range_end are adjusted for the output field.
  // Hence so is the corresponding loop index i.
  // 4. bdry_layer is used to specify the number of cells of output
  // field near the boundary for which some kind of one_sided derivative
  // should be used. Note it bdry_layer depends on input stagger.
  // bdry_layer is anyway calculated, but is only effective at true
  // boundaries.
  // 5. mod_lower[2] = { mod_mult, mod_div }. These are relevant only in
  // the case of evaluating an output field exactly on the boundary. So
  // if the input field is staggered, or if an output field is evaluated
  // at a cell in bdry_layer other than the immediate cell on the
  // boundary, mod_bulk should be used.
  // 6. diff_2nd and diff_2nd_side all take in the address of the field
  // with smallest index. At upper boundary though, for one_sided diff,
  // it is the field with highest index that's passed in, so that the
  // same diff_side function can be reused with delta being negative.
  // Same applies to 4th order.
  // 7. cached values are of length blocksize + 2 * guard. It is
  // important to note which is the first value cached given any result
  // range ( i.e. the range in which results are evaluated ). For bulk,
  // the first value cached is the leftmost value in the "guard cells"
  // of the result range. In the code, i - _order/2 is used to refer to
  // this cell. For lower and upper boundaries, it is always the first
  // cell beyond the boundary that is cached, regardless of guard. The
  // need to include one extra cell is because in higher than 2nd order
  // diff schemes at the lower boundary, one-sided diff for staggered
  // input fields will need the value right on the boundary. For upper
  // boundary, the first cached value is actually unnecessary; the
  // stagger we work with guarantees that at upper boundary, no values
  // beyond the boundary is needed. But this is the convention.
  // 8. there is a variable called range when caching values on
  // boundaries. It is equal to the largest possible number of input
  // fields that is needed to evaluate one-sided derivative. Note this
  // number doesn't any input field beyond the boundary. In other words,
  // this range is equal to the number of input fields when result is
  // evaluated right on the boundary. I don't know why it is range+2
  // number of fields points cached; seems that range+1 or even just
  // range is sufficient. 9.[Problem] cached values at boundary seems a
  // bit strange. The implementation actually keeps caching for each one
  // of the boundary cell, which seems unnecessary. This is not a
  // problem for 2nd order.
  // 10. range_start in transverse directions is modified based on the
  // staggeredness of field in that direction, no matter whether it's at
  // true lower boundary or not. This is the right behavior if at the
  // true boundary in that transverse direction. In bulks, this value
  // will be equal to and eventually be overwritten by the
  // sendGuardCells values.
  if (dir > 2) {
    std::cerr << "Taking derivative in an invalid direction."
              << std::endl;
    return;
  }

  auto& mesh = grid.mesh();

  // First try to obtain the range of the loop from given start and
  // extent parameters
  Vec3<int> range_start{std::max(mesh.guard[0], start[0]),
                        std::max(mesh.guard[1], start[1]),
                        std::max(mesh.guard[2], start[2])};
  Vec3<int> range_end{
      std::min(mesh.guard[0] + mesh.reduced_dim(0), start[0] + ext[0]),
      std::min(mesh.guard[1] + mesh.reduced_dim(1), start[1] + ext[1]),
      std::min(mesh.guard[2] + mesh.reduced_dim(2), start[2] + ext[2])};

  // Modify the range depending on stagger. On directions
  // perpendicular to derivative direction, we include an extra cell
  // if the stagger in that direction is 1. On direction of the
  // derivative, we do the opposite because the result has different
  // stagger from the input stagger. We also obtain the transverse
  // directions on the side.
  unsigned int trans[2]{(dir + 1) % 3, (dir + 2) % 3};
  if (trans[0] < grid.dim())
    range_start[trans[0]] -= (int)params.stagger[trans[0]];
  if (trans[1] < grid.dim())
    range_start[trans[1]] -= (int)params.stagger[trans[1]];
  if (params.boundary[0])
    range_start[dir] -= (int)flip(params.stagger[dir]);

  // Determine the thickness of boundary layer
  // When stagger[dir] = 1 and at least one end assumes zero,
  // it still needs special treatment.
  // FIXME: maintain bndry_layers separately for lower and upper?
  int bndy_layer = 0;
  if (params.order == 2) bndy_layer = 1 - params.stagger[dir];
  //          * static_cast<int>( !( mod.assume_zero[0] ) && !(
  //          mod.assume_zero[1] ) );
  else if (params.order == 4)
    bndy_layer = 2 - params.stagger[dir];
  //          * static_cast<int>( !( mod.assume_zero[0] ) && !(
  //          mod.assume_zero[1] ) );

  // We need the stagger configuration of the result to compute
  // mod_div
  Stagger_t stagger_result = params.stagger;
  stagger_result[dir] = flip(params.stagger[dir]);

  // This is the buffer array for one derivative in a block

  // Outer loop goes over transverse directions. Note: The indices i,
  // j, k correspond to the indices of the result field point.
  // #pragma omp parallel
  // {
  // for collapse(2)
  Scalar* values = new Scalar[block_size + 2 * mesh.guard[dir]];
  // if (omp_get_thread_num() == 0)
  //   std::cout << "Openmp thread number is " << omp_get_num_threads()
  //   << std::endl;

  // #pragma omp for collapse (2)
  for (int k = range_start[trans[0]]; k < range_end[trans[0]]; k++) {
    for (int j = range_start[trans[1]]; j < range_end[trans[1]]; j++) {
      // Can't use loop index to deal with boundary in transverse
      // direction, need to check for singularities

      // This is the transverse index
      int transIdx = j * mesh.idx_increment(trans[1]) +
                     k * mesh.idx_increment(trans[0]);

      // Inner loop goes over the derivative direction. Take
      // block_size strides to minimize memory operations.

      int i = range_start[dir];
      char mod_mult = params.mod_lower[0],
           mod_div = params.mod_lower[1], factor = params.factor[0];
      ///////////////////////////////////////////////////////////////////////
      // Lower boundary. Since i was defined and initialized above,
      // skip initialization
      ///////////////////////////////////////////////////////////////////////
      for (; i <
             range_start[dir] + (bndy_layer * (int)params.boundary[0]);
           i++) {
        if (i > range_start[dir] || params.stagger[dir] != 0) {
          mod_mult = params.mod_bulk[0];
          mod_div = params.mod_bulk[1];
          factor = params.factor[1];
        }
        // Logger::print(0, "Lower boundary,", bndy_layer, "layers");

        int range = 0;
        if (params.order == 2)
          range = 3;
        else if (params.order == 4)
          range = 5;

        for (int n = 0; n < range + 2; n++) {
          int idx =
              (i + n - params.stagger[dir]) * mesh.idx_increment(dir) +
              transIdx;
          values[n] = input[idx];
          multiply_mod(mod_mult, values[n], grid, idx, params.stagger);
        }

        // adjust values[n] for staggered input field when zero is
        // assumed.
        //          if( i == range_start[dir] && stagger[dir] == 1 &&
        //          mod.assume_zero[0] )
        //              values[0] = 0.0;

        // if (i == 2)
        //   Logger::print(0, values[0], values[1], values[2],
        //   values[3], values[4], values[5]);

        Scalar result = 0.0;
        // This is the index for the output field
        int outIdx = i * mesh.idx_increment(dir) + transIdx;
        if (params.order == 2) {
          //            if ( stagger[dir] == 0) {
          result = factor * diff_2nd_side(
                                values + 1 - params.stagger[dir],
                                mesh.delta[dir], params.assume_zero[0]);
          //            } else {
          //              result = factor * diff_2nd( values + 1
          //              -stagger[dir], mesh.delta[dir]);
          //            }
        } else if (params.order == 4) {
          if (i == range_start[dir] && params.stagger[dir] == 0) {
            result = factor *
                     diff_4th_side_mod(values + 1 - params.stagger[dir],
                                       mesh.delta[dir],
                                       params.assume_zero[0]);
          } else {
            result = factor * diff_4th_side(values, mesh.delta[dir]);
          }
        }
        divide_mod(mod_div, result, grid, outIdx, stagger_result);
        output[outIdx] += result;
      }

      // Logger::print(0, "After lower boundary, i is now", i);
      // Logger::print(0, "Doing Bulk until", range_end[dir] -
      // (bndy_layer * (int)upper));
      int bulk_end =
          range_end[dir] - (bndy_layer * (int)params.boundary[1]);
      // Set modifier to be bulk
      mod_mult = params.mod_bulk[0];
      mod_div = params.mod_bulk[1];
      factor = params.factor[1];
      ///////////////////////////////////////////////////////////////////////
      // Bulk
      // Again skip initialization because we reuse the same index
      ///////////////////////////////////////////////////////////////////////
      for (; i < bulk_end; i += block_size) {
        // Fill the buffer array with values first
        for (int n = 0; n < block_size + 2 * mesh.guard[dir]; n++) {
          // if (i - mesh.g)
          // FIXME: _order / 2 is a good indicator?
          int n_i = i - params.order / 2 + n + 1 - params.stagger[dir];
          if (n_i > mesh.dims[dir] - 1) continue;
          // printf ("Exceeding limit: i is %d, n is %d!\n", i, n);
          int idx = n_i * mesh.idx_increment(dir) + transIdx;
          if (idx >= input.size())
            Logger::err("Exceeding boundary at i =", i, "n = ", n);
          else if (idx < 0)
            Logger::err("index is smaller than zero at i =", i,
                        "n = ", n);
          values[n] = input[idx];
          multiply_mod(mod_mult, values[n], grid, idx, params.stagger);
        }
        for (int n = 0; n < block_size; n++) {
          if (i + n >= bulk_end) break;
          // This is the index for the output field
          int outIdx = (i + n) * mesh.idx_increment(dir) + transIdx;
          // if (outIdx >= mesh.size())
          //   printf("!!!Grid size exceeded at i = %d, n = %d\n", i,
          //   n);
          // output[outIdx] = 0.0;
          Scalar result = 0.0;
          if (params.order == 2)
            result = factor * diff_2nd(values + n, mesh.delta[dir]);
          else if (params.order == 4)
            result = factor * diff_4th(values + n, mesh.delta[dir]);
          divide_mod(mod_div, result, grid, outIdx, stagger_result);
          output[outIdx] += result;
          // Logger::print(0, "Processed one point in bulk");
        }
      }

      i = bulk_end;

      ///////////////////////////////////////////////////////////////////////
      // Upper boundary
      ///////////////////////////////////////////////////////////////////////
      for (; i < range_end[dir]; i++) {
        if (i == range_end[dir] - 1 && params.stagger[dir] == 0) {
          // Set modifiers to the upper boundary one
          mod_mult = params.mod_upper[0];
          mod_div = params.mod_upper[1];
          factor = params.factor[2];
        }
        // Logger::print(0, "Doing upper boundary");
        int range = 0;
        if (params.order == 2)
          range = 3;
        else if (params.order == 4)
          range = 5;

        for (int n = 0; n < range + 2; n++) {
          int idx = (i + 1 - params.stagger[dir] - n) *
                        mesh.idx_increment(dir) +
                    transIdx;
          values[n] = input[idx];
          multiply_mod(mod_mult, values[n], grid, idx, params.stagger);
        }

        // adjust values[n] for staggered input field when zero is
        // assumed.
        //          if( i == range_end[dir] - 1 && stagger[dir] == 1 &&
        //          mod.assume_zero[1] )
        //              values[0] = 0.0;

        Scalar result = 0.0;
        // This is the index for the output field
        int outIdx = i * mesh.idx_increment(dir) + transIdx;
        if (params.order == 2) {
          //            if ( stagger[dir] == 0 ) {
          result =
              factor * diff_2nd_side(values + 1 - params.stagger[dir],
                                     -mesh.delta[dir],
                                     params.assume_zero[1]);
          //            } else {
          //              result = factor * diff_2nd( values + 1 -
          //              stagger[dir], -mesh.delta[dir]);
          //            }
        } else if (params.order == 4) {
          if (i == range_end[dir] - 1 && params.stagger[dir] == 0)
            result = factor *
                     diff_4th_side_mod(values + 1 - params.stagger[dir],
                                       -mesh.delta[dir],
                                       params.assume_zero[1]);
          else
            result = factor * diff_4th_side(values, -mesh.delta[dir]);
        }
        divide_mod(mod_div, result, grid, outIdx, stagger_result);
        output[outIdx] += result;
      }
    }
  }

  delete[] values;
}

void
FiniteDiff::derivative(const Grid& grid, const array_type& input,
                       array_type& output, unsigned int dir,
                       const Aperture::diff_params& params) {
  auto mesh = grid.mesh();
  auto start = Index(mesh.guard[0], mesh.guard[1], mesh.guard[2]);
  auto ext = Extent(mesh.reduced_dim(0), mesh.reduced_dim(1),
                    mesh.reduced_dim(2));

  derivative(grid, input, output, dir, start, ext, params);
}

void
FiniteDiff::derivative_bulk(const Grid& grid, const array_type& input,
                            array_type& output, unsigned int dir,
                            const Index& start, const Extent& ext,
                            double factor, bool stagger, char mod_mult,
                            char mod_div, int order) {
  auto& mesh = grid.mesh();
  unsigned int trans[2]{(dir + 1) % 3, (dir + 2) % 3};
  auto stagger_result = flip(stagger);

  Scalar* values = new Scalar[block_size + 2 * mesh.guard[dir]];

  for (int k = start[trans[0]]; k < start[trans[0]] + ext[trans[0]];
       k++) {
    for (int j = start[trans[1]]; j < start[trans[1]] + ext[trans[1]];
         j++) {
      // Can't use loop index to deal with boundary in transverse
      // direction, need to check for singularities

      // This is the transverse index
      int transIdx = j * mesh.idx_increment(trans[1]) +
                     k * mesh.idx_increment(trans[0]);

      for (int i = start[dir]; i < start[dir] + ext[dir];
           i += block_size) {
        // Fill the buffer array with values first
        for (int n = 0; n < block_size + 2 * mesh.guard[dir]; n++) {
          // FIXME: order / 2 is a good indicator?
          int n_i = i - order / 2 + n + 1 - stagger;
          if (n_i >= mesh.dims[dir]) continue;
          // printf ("Exceeding limit: i is %d, n is %d!\n", i, n);
          int idx = n_i * mesh.idx_increment(dir) + transIdx;
          if (idx >= input.size())
            Logger::err("Exceeding boundary at i =", i, "n = ", n,
                        "idx =", idx);
          else if (idx < 0)
            Logger::err("index is smaller than zero at i =", i,
                        "n = ", n, "idx = ", idx);
          values[n] = input[idx];
          multiply_mod(mod_mult, values[n], grid, idx, stagger);
        }
        for (int n = 0; n < block_size; n++) {
          if (i + n >= start[dir] + ext[dir]) break;
          // This is the index for the output field
          int outIdx = (i + n) * mesh.idx_increment(dir) + transIdx;
          // if (outIdx >= mesh.size())
          //   printf("!!!Grid size exceeded at i = %d, n = %d\n", i,
          //   n);
          // output[outIdx] = 0.0;
          Scalar result = 0.0;
          if (order == 2)
            result = factor * diff_2nd(values + n, mesh.delta[dir]);
          else if (order == 4)
            result = factor * diff_4th(values + n, mesh.delta[dir]);
          divide_mod(mod_div, result, grid, outIdx, stagger_result);
          output[outIdx] += result;
          // Logger::print(0, "Processed one point in bulk");
        }
      }
    }
  }

  delete[] values;
}

void
FiniteDiff::derivative_boundary(const Grid& grid,
                                const array_type& input,
                                array_type& output, unsigned int dir,
                                const Index& start, const Extent& ext,
                                double factor, bool stagger,
                                char mod_mult, char mod_div, int order,
                                int side) {}

void
FiniteDiff::compute_curl(const vfield& input, vfield& output,
                         const bool* isBoundary,
                         const Aperture::Index& start,
                         const Aperture::Extent& ext, int order) {
  auto& grid = input.grid();
  for (unsigned int i = 0; i < VECTOR_DIM; i++) {
    output.data(i).assign(0.0);
    // (Curl F)_i = D_j F_k - D_k F_j
    unsigned int j = (i + 1) % VECTOR_DIM;
    unsigned int k = (i + 2) % VECTOR_DIM;

    // First find the stagger of the input field type
    auto stagger_j = input.stagger(j);
    auto stagger_k = input.stagger(k);

    diff_params mod;
    // Compute D_j F_k, dealing with singularity
    if (grid.dim() > j) {
      mod.mod_bulk[0] = mod.mod_lower[0] = mod.mod_upper[0] = m_h[k];
      mod.mod_bulk[1] = mod.mod_lower[1] = mod.mod_upper[1] =
          m_h[j] | m_h[k];  // We can also use +?
      // mod.assume_zero =
      if (j == 1 && k != 0 &&
          (grid.type() == MetricType::spherical ||
           grid.type() == MetricType::log_spherical)) {
        mod.mod_lower[0] = mod.mod_upper[0] = 0;
        mod.mod_lower[1] = mod.mod_upper[1] = m_h[j];
        mod.factor[0] = mod.factor[2] = 2.0;
        mod.assume_zero[0] = mod.assume_zero[1] = true;
        mod.stagger = stagger_k;
        mod.boundary[0] = isBoundary[2 * j];
        mod.boundary[1] = isBoundary[2 * j + 1];
        mod.order = order;
      }
      derivative(grid, input.data(k), output.data(i), j, start, ext,
                 mod);
    }

    if (grid.dim() > k) {
      // Compute D_k F_j, dealing with singularity
      mod.mod_bulk[0] = mod.mod_lower[0] = mod.mod_upper[0] = m_h[j];
      mod.mod_bulk[1] = mod.mod_lower[1] = mod.mod_upper[1] =
          m_h[j] | m_h[k];  // We can also use +?
      mod.factor[0] = mod.factor[1] = mod.factor[2] = -1.0;
      // if (k == 1 && j != 0 && (_scales -> Type() ==
      // CoordType::SPHERICAL || _scales -> Type() ==
      // CoordType::LOG_SPHERICAL)) {
      if (k == 1 && j != 0 &&
          (grid.type() == MetricType::spherical ||
           grid.type() == MetricType::log_spherical)) {
        mod.mod_lower[0] = mod.mod_upper[0] = 0;
        mod.mod_lower[1] = mod.mod_upper[1] = m_h[k];
        mod.factor[0] = mod.factor[2] = -2.0;
        mod.assume_zero[0] = mod.assume_zero[1] = true;
        mod.stagger = stagger_j;
        mod.boundary[0] = isBoundary[2 * k];
        mod.boundary[1] = isBoundary[2 * k + 1];
        mod.order = order;
      }
      // Derivative(input.data(j), output.data(i), k, stagger_j, start,
      // ext,
      //            isBoundary[2*k], isBoundary[2*k + 1], mod);
      derivative(grid, input.data(j), output.data(i), k, start, ext,
                 mod);
    }
  }
}

void
FiniteDiff::compute_curl(const vfield& input, vfield& output,
                         const bool* isBoundary, int order) {
  auto& grid = input.grid();
  auto mesh = grid.mesh();
  auto start = Index(mesh.guard[0], mesh.guard[1], mesh.guard[2]);
  auto ext = Extent(mesh.reduced_dim(0), mesh.reduced_dim(1),
                    mesh.reduced_dim(2));

  compute_curl(input, output, isBoundary, start, ext, order);
}

void
FiniteDiff::compute_divergence(const vfield& input, sfield& output,
                               const bool isBoundary[],
                               const Index& start, const Extent& ext,
                               int order) {
  auto& grid = input.grid();
  output.assign(0.0);

  // Only loop over the first two dimensions if _dim is 2
  for (unsigned int i = 0; i < grid.dim(); i++) {
    auto stagger = input.stagger(i);

    int j = (i + 1) % VECTOR_DIM;
    int k = (i + 2) % VECTOR_DIM;

    diff_params mod;

    // Compute D_i F_i, dealing with singularity
    mod.mod_bulk[0] = mod.mod_lower[0] = mod.mod_upper[0] =
        m_h[j] | m_h[k];
    mod.mod_bulk[1] = mod.mod_lower[1] = mod.mod_upper[1] =
        m_h[i] | m_h[j] | m_h[k];

    // if (i == 1 && (_scales -> Type() == CoordType::SPHERICAL ||
    // _scales -> Type() == CoordType::LOG_SPHERICAL)) {
    if (i == 1 && (grid.type() == MetricType::spherical ||
                   grid.type() == MetricType::log_spherical)) {
      mod.mod_lower[0] = mod.mod_upper[0] = 0;
      mod.mod_lower[1] = mod.mod_upper[1] = m_h[i];
      mod.factor[0] = mod.factor[2] = 2.0;
      mod.assume_zero[0] = mod.assume_zero[1] = true;
      mod.stagger = stagger;
      mod.boundary[0] = isBoundary[2 * i];
      mod.boundary[1] = isBoundary[2 * i + 1];
      mod.order = order;
    }
    derivative(grid, input.data(i), output.data(), i, start, ext, mod);
  }
}

void
FiniteDiff::compute_divergence(const vfield& input, sfield& output,
                               const bool* isBoundary, int order) {
  auto& grid = input.grid();
  auto mesh = grid.mesh();
  auto start = Index(mesh.guard[0], mesh.guard[1], mesh.guard[2]);
  auto ext = Extent(mesh.reduced_dim(0), mesh.reduced_dim(1),
                    mesh.reduced_dim(2));

  compute_divergence(input, output, isBoundary, start, ext, order);
}

void
FiniteDiff::compute_gradient(const sfield& input, vfield& output,
                             const bool* isBoundary,
                             const Aperture::Index& start,
                             const Aperture::Extent& ext, int order) {
  auto& grid = input.grid();
  output.assign(0.0);
  auto stagger = input.stagger();

  // Only loop over the first two dimensions if _dim is 2
  for (unsigned int i = 0; i < grid.dim(); i++) {
    diff_params mod;

    mod.mod_bulk[0] = mod.mod_lower[0] = mod.mod_upper[0] = 0;
    mod.mod_bulk[1] = mod.mod_lower[1] = mod.mod_upper[1] = m_h[i];
    mod.stagger = stagger;
    mod.boundary[0] = isBoundary[2 * i];
    mod.boundary[1] = isBoundary[2 * i + 1];
    mod.order = order;
    //    if (i == 1 && (Coord == CoordType::SPHERICAL || Coord ==
    //    CoordType::LOG_SPHERICAL)) {
    //      mod.assume_zero[0] = mod.assume_zero[1] = true;
    //    }
    // TODO: in 3D, deal with 1/h_phi at theta boundaries??

    derivative(grid, input.data(), output.data(i), i, start, ext, mod);
  }
}
