#ifndef _MACRO_TRICKERY_H_
#define _MACRO_TRICKERY_H_

#include "cuda/cuda_control.h"
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/facilities/expand.hpp>
#include <boost/preprocessor/seq/fold_right.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/to_tuple.hpp>

////////////////////////////////////////////////////////////////////
// These are helper macros for defining the particle data structure.
////////////////////////////////////////////////////////////////////

#define ESC(...) __VA_ARGS__

#define EXPAND_ELEMS(macro, elem)        \
  macro(BOOST_PP_TUPLE_ELEM(3, 0, elem), \
        BOOST_PP_TUPLE_ELEM(3, 1, elem), \
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

#define ASSIGN_ENTRY_(arr1, idx1, arr2, idx2, name) \
  arr1.name[idx1] = arr2.name[idx2];
#define ASSIGN_ENTRY(r, data, elem)                           \
  ASSIGN_ENTRY_(                                              \
      BOOST_PP_SEQ_ELEM(0, data), BOOST_PP_SEQ_ELEM(1, data), \
      BOOST_PP_SEQ_ELEM(2, data), BOOST_PP_SEQ_ELEM(3, data), \
      BOOST_PP_TUPLE_ELEM(3, 1, elem))

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

////////////////////////////////////////////////////////////////////

#define DEF_PARTICLE_STRUCT(name, content)                             \
  namespace Aperture {                                                 \
  struct single_##name##_t {                                           \
    BOOST_PP_SEQ_FOR_EACH(DEF_ENTRY, _,                                \
                          GLK_PP_SEQ_DOUBLE_PARENS(content))           \
  };                                                                   \
                                                                       \
  struct name##_data {                                                 \
    BOOST_PP_SEQ_FOR_EACH(DEF_PTR_ENTRY, _,                            \
                          GLK_PP_SEQ_DOUBLE_PARENS(content))           \
    single_##name##_t operator[](size_t idx) const;                    \
    enum {                                                             \
      size = BOOST_PP_SEQ_FOLD_RIGHT(                                  \
          ADD_SIZEOF, 0,                                               \
          BOOST_PP_SEQ_FOR_EACH(GET_TYPE, _,                           \
                                GLK_PP_SEQ_DOUBLE_PARENS(content)))    \
    };                                                                 \
  };                                                                   \
                                                                       \
  template <>                                                          \
  struct ptc_array_type<single_##name##_t> {                           \
    typedef name##_data type;                                          \
  };                                                                   \
                                                                       \
  HD_INLINE void assign_ptc(name##_data array_1, size_t idx_1,         \
                            name##_data array_2, size_t idx_2) {       \
    BOOST_PP_SEQ_FOR_EACH(ASSIGN_ENTRY,                                \
                          (array_1)(idx_1)(array_2)(idx_2),            \
                          GLK_PP_SEQ_DOUBLE_PARENS(content))           \
  }                                                                    \
  }                                                                    \
  VISITABLE_STRUCT(                                                    \
      Aperture::single_##name##_t,                                     \
      BOOST_PP_EXPAND(ESC BOOST_PP_SEQ_TO_TUPLE(BOOST_PP_SEQ_FOR_EACH( \
          GET_NAME, _, GLK_PP_SEQ_DOUBLE_PARENS(content)))));          \
  VISITABLE_STRUCT(                                                    \
      Aperture::name##_data,                                           \
      BOOST_PP_EXPAND(ESC BOOST_PP_SEQ_TO_TUPLE(BOOST_PP_SEQ_FOR_EACH( \
          GET_NAME, _, GLK_PP_SEQ_DOUBLE_PARENS(content)))))

#endif  // _MACRO_TRICKERY_H_
