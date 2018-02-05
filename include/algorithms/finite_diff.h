#ifndef _FINITE_DIFF_H_
#define _FINITE_DIFF_H_

#include "data/fields.h"
#include "data/multi_array.h"
#include "sim_environment.h"

namespace Aperture {

struct diff_params {
  int order = 2;
  char mod_lower[2] = {0, 0};  // { mult_mod, div_mod }
  char mod_bulk[2] = {0, 0};
  char mod_upper[2] = {0, 0};
  Scalar factor[3] = {1.0, 1.0, 1.0};
  bool assume_zero[2] = {false, false};
  Stagger_t stagger = Stagger_t("000");
  bool boundary[2] = {false, false};
  bool one_sided[2] = {false, false};
};

class FiniteDiff {
 private:
  typedef MultiArray<Scalar> array_type;
  typedef VectorField<Scalar> vfield;
  typedef ScalarField<Scalar> sfield;

  // const Grid& m_grid;

  // Auxiliary array used to turn on/off bits for each scale function
  static constexpr int m_h[3] = {1, 2, 4};

  static void derivative_bulk(const Grid& grid, const array_type& input,
                              array_type& output, unsigned int dir,
                              const Index& start, const Extent& ext,
                              double factor, bool stagger, char mod_mult,
                              char mod_div, int order);

  static void derivative_boundary(const Grid& grid, const array_type& input,
                                  array_type& output, unsigned int dir,
                                  const Index& start, const Extent& ext,
                                  double factor, bool stagger, char mod_mult,
                                  char mod_div, int order, int side);

 public:
  // FiniteDiff(const Grid& grid);
  FiniteDiff();
  ~FiniteDiff();

  static void derivative(const Grid& grid, const array_type& input,
                         array_type& output, unsigned int dir,
                         const diff_params& params);
  static void derivative(const Grid& grid, const array_type& input,
                         array_type& output, unsigned int dir,
                         const Index start, const Extent ext,
                         const diff_params& params);

  static void compute_curl(const vfield& input, vfield& output,
                           const bool isBoundary[], const Index& start,
                           const Extent& ext, int order = 2);
  static void compute_curl(const vfield& input, vfield& output,
                           const bool isBoundary[], int order = 2);

  static void compute_divergence(const vfield& input, sfield& output,
                                 const bool isBoundary[], const Index& start,
                                 const Extent& ext, int order = 2);

  static void compute_divergence(const vfield& input, sfield& output,
                                 const bool isBoundary[], int order = 2);

  static void compute_gradient(const sfield& input, vfield& output,
                               const bool isBoundary[], const Index& start,
                               const Extent& ext, int order = 2);

  static void compute_gradient(const sfield& input, vfield& output,
                               const bool isBoundary[], int order = 2);

  // friend class testFD;
};  // ----- end of class finite_diff -----
}

#endif  // _FINITE_DIFF_H_
