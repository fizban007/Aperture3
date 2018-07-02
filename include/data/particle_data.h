#ifndef _PARTICLE_DATA_H_
#define _PARTICLE_DATA_H_

// #include "boost/fusion/container/vector.hpp"
// #include "boost/fusion/include/adapt_struct.hpp"
// #include "boost/fusion/include/for_each.hpp"
// #include "boost/fusion/include/zip_view.hpp"
#include "constant_defs.h"
#include "data/typedefs.h"
#include "data/vec3.h"
#include "cuda/cuda_control.h"
#include "visit_struct/visit_struct.hpp"
// #include "data/vec4.h"
#include <cinttypes>
#include <type_traits>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/facilities/expand.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/to_tuple.hpp>
#include <boost/preprocessor/seq/fold_right.hpp>
#include <cinttypes>

#define ESC(...) __VA_ARGS__

#define EXPAND_ELEMS(macro, elem)               \
  macro(BOOST_PP_TUPLE_ELEM(3, 0, elem),        \
        BOOST_PP_TUPLE_ELEM(3, 1, elem),        \
        BOOST_PP_TUPLE_ELEM(3, 2, elem))

#define DEF_ENTRY_(type, name, dv) type name = (dv);
#define DEF_ENTRY(r, data, elem) EXPAND_ELEMS(DEF_ENTRY_, elem)

#define DEF_PTR_ENTRY_(type, name, dv) type* name;
#define DEF_PTR_ENTRY(r, data, elem) EXPAND_ELEMS(DEF_PTR_ENTRY_, elem)

#define GET_NAME_(type, name, dv) (name)
#define GET_NAME(r, data, elem) EXPAND_ELEMS(GET_NAME_, elem)
#define GET_TYPE_(type, name, dv) (type)
#define GET_TYPE(r, data, elem) EXPAND_ELEMS(GET_TYPE_, elem)
#define GET_TYPE_NAME_(type, name, dv) (type, name)
#define GET_TYPE_NAME(r, data, elem) EXPAND_ELEMS(GET_TYPE_NAME_, elem)
#define GET_PTR_NAME_(type, name, dv) (type*, name)
#define GET_PTR_NAME(r, data, elem) EXPAND_ELEMS(GET_PTR_NAME_, elem)

#define ADD_SIZEOF(s, state, elem) state + sizeof(elem)

#define GLK_PP_DETAIL_SEQ_DOUBLE_PARENS_0(...) \
     ((__VA_ARGS__)) GLK_PP_DETAIL_SEQ_DOUBLE_PARENS_1

#define GLK_PP_DETAIL_SEQ_DOUBLE_PARENS_1(...) \
     ((__VA_ARGS__)) GLK_PP_DETAIL_SEQ_DOUBLE_PARENS_0

#define GLK_PP_DETAIL_SEQ_DOUBLE_PARENS_0_END
#define GLK_PP_DETAIL_SEQ_DOUBLE_PARENS_1_END

// Double the parentheses of a Boost.PP sequence
// I.e. (a, b)(c, d) becomes ((a, b))((c, d))
#define GLK_PP_SEQ_DOUBLE_PARENS(seq) \
    BOOST_PP_CAT(GLK_PP_DETAIL_SEQ_DOUBLE_PARENS_0 seq, _END)

#define DEF_PARTICLE_STRUCT(name, content)                                 \
  namespace Aperture {                                                  \
  struct single_ ## name ## _t {                                        \
    BOOST_PP_SEQ_FOR_EACH(DEF_ENTRY, _,                                 \
                          GLK_PP_SEQ_DOUBLE_PARENS(content))            \
  };                                                                    \
                                                                        \
  struct name ## _data {                                                \
    BOOST_PP_SEQ_FOR_EACH(DEF_PTR_ENTRY, _,                             \
                          GLK_PP_SEQ_DOUBLE_PARENS(content))            \
    single_ ## name ## _t operator[](size_t idx) const;     \
    enum {                                                              \
      size = BOOST_PP_SEQ_FOLD_RIGHT(ADD_SIZEOF, 0,                     \
                                     BOOST_PP_SEQ_FOR_EACH(GET_TYPE, _, GLK_PP_SEQ_DOUBLE_PARENS(content))) \
    };                                                                  \
  };                                                                    \
                                                                        \
  template <>                                                           \
  struct particle_array_type<single_ ## name ## _t> {                   \
    typedef name ## _data type;                                              \
  };                                                                    \
  }                                                                     \
  VISITABLE_STRUCT(Aperture::single_ ## name ## _t,                  \
                   BOOST_PP_EXPAND(ESC BOOST_PP_SEQ_TO_TUPLE(        \
                       BOOST_PP_SEQ_FOR_EACH(GET_NAME, _, GLK_PP_SEQ_DOUBLE_PARENS(content))))); \
  VISITABLE_STRUCT(Aperture::name ## _data,                          \
                   BOOST_PP_EXPAND(ESC BOOST_PP_SEQ_TO_TUPLE(        \
                       BOOST_PP_SEQ_FOR_EACH(GET_NAME, _, GLK_PP_SEQ_DOUBLE_PARENS(content)))))
  // BOOST_FUSION_ADAPT_STRUCT(Aperture::single_ ## name ## _t,                  \
  //                           BOOST_PP_SEQ_FOR_EACH(GET_TYPE_NAME, _, GLK_PP_SEQ_DOUBLE_PARENS(content))); \
  // BOOST_FUSION_ADAPT_STRUCT(Aperture::name ## _data,                          \
  //                      BOOST_PP_SEQ_FOR_EACH(GET_PTR_NAME, _, GLK_PP_SEQ_DOUBLE_PARENS(content)))

namespace Aperture {
  template <typename SingleType>
  struct particle_array_type;
}

DEF_PARTICLE_STRUCT(particle,
                    (Aperture::Pos_t, x1, 0.0)
                    (Aperture::Pos_t, x2, 0.0)
                    (Aperture::Pos_t, x3, 0.0)
                    (Aperture::Scalar, p1, 0.0)
                    (Aperture::Scalar, p2, 0.0)
                    (Aperture::Scalar, p3, 0.0)
                    (Aperture::Scalar, weight, 0.0)
                    (uint32_t, cell, MAX_CELL)
                    (uint32_t, tile, MAX_TILE)
                    (uint32_t, flag, 0)
                    );

DEF_PARTICLE_STRUCT(photon,
                    (Aperture::Pos_t, x1, 0.0)
                    (Aperture::Pos_t, x2, 0.0)
                    (Aperture::Pos_t, x3, 0.0)
                    (Aperture::Scalar, p1, 0.0)
                    (Aperture::Scalar, p2, 0.0)
                    (Aperture::Scalar, p3, 0.0)
                    (Aperture::Scalar, weight, 0.0)
                    (Aperture::Scalar, path_left, 0.0)
                    (uint32_t, cell, MAX_CELL)
                    (uint32_t, tile, MAX_TILE)
                    (uint32_t, flag, 0)
                    );

// namespace Aperture {

// struct single_particle_t {
//   Pos_t x1 = 0.0;
//   Pos_t x2 = 0.0;
//   Pos_t x3 = 0.0;
//   Scalar p1 = 0.0;
//   Scalar p2 = 0.0;
//   Scalar p3 = 0.0;
//   Scalar weight = 0.0;
//   // Defulat MAX_CELL means empty particle slot
//   uint32_t cell = MAX_CELL;
//   uint32_t tile = 0;
//   uint32_t flag = 0;

//   HOST_DEVICE single_particle_t() {}
// };

// struct single_photon_t {
//   Pos_t x1 = 0.0;
//   Pos_t x2 = 0.0;
//   Pos_t x3 = 0.0;
//   Scalar p1 = 0.0;
//   Scalar p2 = 0.0;
//   Scalar p3 = 0.0;
//   Scalar path_left = 0.0;
//   Scalar weight = 0.0;
//   // Defulat MAX_CELL means empty particle slot
//   uint32_t cell = MAX_CELL;
//   uint32_t tile = 0;
//   uint32_t flag = 0;

//   HOST_DEVICE single_photon_t() {}
// };
// }

// VISITABLE_STRUCT(Aperture::single_particle_t, x1, x2, x3, p1, p2, p3,
//                  weight, cell, tile, flag);

// VISITABLE_STRUCT(Aperture::single_photon_t, x1, x2, x3, p1, p2, p3,
//                  path_left, weight, cell, tile, flag);

// BOOST_FUSION_ADAPT_STRUCT(Aperture::single_particle_t,
//                           (Aperture::Pos_t, x1)
//                           (Aperture::Pos_t, x2)
//                           (Aperture::Pos_t, x3)
//                           (Aperture::Scalar, p1)
//                           (Aperture::Scalar, p2)
//                           (Aperture::Scalar, p3)
//                           (Aperture::Scalar, weight)
//                           (uint32_t, cell)
//                           (uint32_t, tile)
//                           (uint32_t, flag));

// BOOST_FUSION_ADAPT_STRUCT(Aperture::single_photon_t,
//                           (Aperture::Pos_t, x1)
//                           (Aperture::Pos_t, x2)
//                           (Aperture::Pos_t, x3)
//                           (Aperture::Scalar, p1)
//                           (Aperture::Scalar, p2)
//                           (Aperture::Scalar, p3)
//                           (Aperture::Scalar, path_left)
//                           (Aperture::Scalar, weight)
//                           (uint32_t, cell)
//                           (uint32_t, tile)
//                           (uint32_t, flag));

// namespace Aperture {

// struct particle_data {
//   // NOTE: This size needs to be exact, otherwise the program
//   // will malfunction due to misallocation of memory

//   // NOTE: This size is also NOT equal to the size of the
//   // single_particle_t struct, due to padding
//   enum {
//     size = sizeof(Pos_t) * 3 + sizeof(Scalar) * 4 + sizeof(uint32_t) * 3
//   };

//   Pos_t* x1;
//   Pos_t* x2;
//   Pos_t* x3;
//   Scalar* p1;
//   Scalar* p2;
//   Scalar* p3;
//   Scalar* weight;

//   uint32_t* cell;
//   uint32_t* tile;
//   uint32_t* flag;

//   HOST_DEVICE particle_data() {}

//   // HOST_DEVICE single_particle_t operator[](size_t idx) const;
// };

// struct photon_data {
//   // NOTE: This size needs to be exact, otherwise the program
//   // will malfunction due to misallocation of memory

//   // NOTE: This size is also NOT equal to the size of the
//   // single_photon_t struct, due to padding
//   enum {
//     size = sizeof(Pos_t) * 3 + sizeof(Scalar) * 5 + sizeof(uint32_t) * 3
//   };

//   Pos_t* x1;
//   Pos_t* x2;
//   Pos_t* x3;
//   Scalar* p1;
//   Scalar* p2;
//   Scalar* p3;
//   Scalar* path_left;
//   Scalar* weight;
//   uint32_t* cell;
//   uint32_t* tile;
//   uint32_t* flag;

//   HOST_DEVICE photon_data() {}

//   // HOST_DEVICE single_photon_t operator[](size_t idx) const;
// };

// template <>
// struct particle_array_type<single_particle_t> {
//   typedef particle_data type;
// };

// template <>
// struct particle_array_type<single_photon_t> {
//   typedef photon_data type;
// };
// }

// VISITABLE_STRUCT(Aperture::particle_data, x1, x2, x3, p1, p2, p3,
//                  weight, cell, tile, flag);

// VISITABLE_STRUCT(Aperture::photon_data, x1, x2, x3, p1, p2, p3,
//                  path_left, weight, cell, tile, flag);

// BOOST_FUSION_ADAPT_STRUCT(Aperture::particle_data,
//                           (Aperture::Pos_t*, x1)
//                           (Aperture::Pos_t*, x2)
//                           (Aperture::Pos_t*, x3)
//                           (Aperture::Scalar*, p1)
//                           (Aperture::Scalar*, p2)
//                           (Aperture::Scalar*, p3)
//                           (Aperture::Scalar*, weight)
//                           (uint32_t*, cell)
//                           (uint32_t*, tile)
//                           (uint32_t*, flag));

// BOOST_FUSION_ADAPT_STRUCT(Aperture::photon_data,
//                           (Aperture::Pos_t*, x1)
//                           (Aperture::Pos_t*, x2)
//                           (Aperture::Pos_t*, x3)
//                           (Aperture::Scalar*, p1)
//                           (Aperture::Scalar*, p2)
//                           (Aperture::Scalar*, p3)
//                           (Aperture::Scalar*, path_left)
//                           (Aperture::Scalar*, weight)
//                           (uint32_t*, cell)
//                           (uint32_t*, tile)
//                           (uint32_t*, flag));

// BOOST_FUSION_ADAPT_STRUCT(Aperture::particle_data, x1, dx1,
//                            p1, gamma, cell, flag);

// BOOST_FUSION_ADAPT_STRUCT(Aperture::photon_data, x1, p1, path_left, cell, flag);

#endif  // _PARTICLE_DATA_H_
