#ifndef _PARTICLE_DATA_H_
#define _PARTICLE_DATA_H_

#include "boost/fusion/container/vector.hpp"
#include "boost/fusion/include/adapt_struct.hpp"
#include "boost/fusion/include/for_each.hpp"
#include "boost/fusion/include/zip_view.hpp"
#include "constant_defs.h"
#include "data/typedefs.h"
#include "data/vec3.h"
#include "cuda/cuda_control.h"
// #include "data/vec4.h"
#include <cinttypes>
#include <type_traits>

namespace Aperture {

struct single_particle_t {
  Pos_t x1 = 0.0;
  Pos_t x2 = 0.0;
  Pos_t x3 = 0.0;
  Scalar p1 = 0.0;
  Scalar p2 = 0.0;
  Scalar p3 = 0.0;
  Scalar weight = 0.0;
  // Defulat MAX_CELL means empty particle slot
  uint32_t cell = MAX_CELL;
  uint32_t tile = 0;
  uint32_t flag = 0;

  HOST_DEVICE single_particle_t() {}
};

struct single_photon_t {
  Pos_t x1 = 0.0;
  Pos_t x2 = 0.0;
  Pos_t x3 = 0.0;
  Scalar p1 = 0.0;
  Scalar p2 = 0.0;
  Scalar p3 = 0.0;
  Scalar path_left = 0.0;
  Scalar weight = 0.0;
  // Defulat MAX_CELL means empty particle slot
  uint32_t cell = MAX_CELL;
  uint32_t tile = 0;
  uint32_t flag = 0;

  HOST_DEVICE single_photon_t() {}
};
}

BOOST_FUSION_ADAPT_STRUCT(Aperture::single_particle_t,
                          (Aperture::Pos_t, x1)
                          (Aperture::Pos_t, x2)
                          (Aperture::Pos_t, x3)
                          (Aperture::Scalar, p1)
                          (Aperture::Scalar, p2)
                          (Aperture::Scalar, p3)
                          (Aperture::Scalar, weight)
                          (uint32_t, cell)
                          (uint32_t, tile)
                          (uint32_t, flag));

BOOST_FUSION_ADAPT_STRUCT(Aperture::single_photon_t,
                          (Aperture::Pos_t, x1)
                          (Aperture::Pos_t, x2)
                          (Aperture::Pos_t, x3)
                          (Aperture::Scalar, p1)
                          (Aperture::Scalar, p2)
                          (Aperture::Scalar, p3)
                          (Aperture::Scalar, path_left)
                          (Aperture::Scalar, weight)
                          (uint32_t, cell)
                          (uint32_t, tile)
                          (uint32_t, flag));

namespace Aperture {

struct particle_data {
  // NOTE: This size needs to be exact, otherwise the program
  // will malfunction due to misallocation of memory

  // NOTE: This size is also NOT equal to the size of the
  // single_particle_t struct, due to padding
  enum {
    size = sizeof(Pos_t) * 3 + sizeof(Scalar) * 4 + sizeof(uint32_t) * 3
  };

  Pos_t* x1;
  Pos_t* x2;
  Pos_t* x3;
  Scalar* p1;
  Scalar* p2;
  Scalar* p3;
  Scalar* weight;

  uint32_t* cell;
  uint32_t* tile;
  uint32_t* flag;

  HOST_DEVICE particle_data() {}

  HOST_DEVICE single_particle_t operator[](size_t idx) const;
};

struct photon_data {
  // NOTE: This size needs to be exact, otherwise the program
  // will malfunction due to misallocation of memory

  // NOTE: This size is also NOT equal to the size of the
  // single_photon_t struct, due to padding
  enum {
    size = sizeof(Pos_t) * 3 + sizeof(Scalar) * 5 + sizeof(uint32_t) * 3
  };

  Pos_t* x1;
  Pos_t* x2;
  Pos_t* x3;
  Scalar* p1;
  Scalar* p2;
  Scalar* p3;
  Scalar* path_left;
  Scalar* weight;
  uint32_t* cell;
  uint32_t* tile;
  uint32_t* flag;

  HOST_DEVICE photon_data() {}

  HOST_DEVICE single_photon_t operator[](size_t idx) const;
};

template <typename SingleType>
struct particle_array_type;

template <>
struct particle_array_type<single_particle_t> {
  typedef particle_data type;
};

template <>
struct particle_array_type<single_photon_t> {
  typedef photon_data type;
};
}

BOOST_FUSION_ADAPT_STRUCT(Aperture::particle_data,
                          (Aperture::Pos_t*, x1)
                          (Aperture::Pos_t*, x2)
                          (Aperture::Pos_t*, x3)
                          (Aperture::Scalar*, p1)
                          (Aperture::Scalar*, p2)
                          (Aperture::Scalar*, p3)
                          (Aperture::Scalar*, weight)
                          (uint32_t*, cell)
                          (uint32_t*, tile)
                          (uint32_t*, flag));

BOOST_FUSION_ADAPT_STRUCT(Aperture::photon_data,
                          (Aperture::Pos_t*, x1)
                          (Aperture::Pos_t*, x2)
                          (Aperture::Pos_t*, x3)
                          (Aperture::Scalar*, p1)
                          (Aperture::Scalar*, p2)
                          (Aperture::Scalar*, p3)
                          (Aperture::Scalar*, path_left)
                          (Aperture::Scalar*, weight)
                          (uint32_t*, cell)
                          (uint32_t*, tile)
                          (uint32_t*, flag));

// BOOST_FUSION_ADAPT_STRUCT(Aperture::particle_data, x1, dx1,
//                            p1, gamma, cell, flag);

// BOOST_FUSION_ADAPT_STRUCT(Aperture::photon_data, x1, p1, path_left, cell, flag);

#endif  // _PARTICLE_DATA_H_
