#ifndef _PARTICLE_DATA_H_
#define _PARTICLE_DATA_H_

#include "boost/fusion/container/vector.hpp"
#include "boost/fusion/include/adapt_struct.hpp"
#include "boost/fusion/include/for_each.hpp"
#include "boost/fusion/include/zip_view.hpp"
#include "constant_defs.h"
#include "data/typedefs.h"
#include "data/vec3.h"
// #include "data/vec4.h"
#include <cinttypes>
#include <type_traits>

namespace Aperture {

struct single_particle_t {
  Pos_t x1 = 0.0;
  Pos_t dx1 = 0.0;
  Scalar p1 = 0.0;
  Scalar gamma = 0.0;
  // Defulat MAX_CELL means empty particle slot
  uint32_t cell = MAX_CELL;
  uint32_t flag = 0;

  // A series of set methods so that one can chain them
  single_particle_t& set_x(Pos_t x) {
    x1 = x;
    return *this;
  }

  single_particle_t& set_dx(Pos_t dx) {
    dx1 = dx;
    return *this;
  }

  single_particle_t& set_p(Scalar p) {
    p1 = p;
    gamma = sqrt(1.0 + p1 * p1);
    return *this;
  }

  single_particle_t& set_cell(uint32_t c) {
    cell = c;
    return *this;
  }

  single_particle_t& set_flag(uint32_t f) {
    flag = f;
    return *this;
  }
};

struct single_photon_t {
  Pos_t x1 = 0.0;
  Scalar p1 = 0.0;
  Scalar path_left = 0.0;
  // Defulat MAX_CELL means empty particle slot
  uint32_t cell = MAX_CELL;
  uint32_t flag = 0;

  // A series of set methods so that one can chain them
  single_photon_t& set_x(Pos_t x) {
    x1 = x;
    return *this;
  }

  single_photon_t& set_p(Scalar p) {
    p1 = p;
    return *this;
  }

  single_photon_t& set_cell(uint32_t c) {
    cell = c;
    return *this;
  }

  single_photon_t& set_path_left(double p) {
    path_left = p;
    return *this;
  }

  single_photon_t& set_flag(uint32_t f) {
    flag = f;
    return *this;
  }
};
}

BOOST_FUSION_ADAPT_STRUCT(Aperture::single_particle_t, x1, dx1,
                           p1, gamma, cell, flag);

BOOST_FUSION_ADAPT_STRUCT(Aperture::single_photon_t, x1, p1, cell, path_left,
                          flag);

namespace Aperture {

struct particle_data {
  // NOTE: This size needs to be exact, otherwise the program
  // will malfunction due to misallocation of memory

  // NOTE: This size is also NOT equal to the size of the
  // single_particle_t struct, due to padding
  enum {
    size = sizeof(Pos_t) * 2 + sizeof(Scalar) * 2 + sizeof(uint32_t) * 2
  };

  Pos_t* x1;
  Pos_t* dx1;
  Scalar* p1;
  Scalar* gamma;

  uint32_t* cell;
  uint32_t* flag;

  single_particle_t operator[](int idx) const;
};

struct photon_data {
  // NOTE: This size needs to be exact, otherwise the program
  // will malfunction due to misallocation of memory

  // NOTE: This size is also NOT equal to the size of the
  // single_photon_t struct, due to padding
  enum {
    size = sizeof(Pos_t) * 1 + sizeof(Scalar) * 2 + sizeof(uint32_t) * 2
  };

  Pos_t* x1;
  Scalar* p1;
  Scalar* path_left;
  uint32_t* cell;
  uint32_t* flag;

  single_photon_t operator[](int idx) const;
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

BOOST_FUSION_ADAPT_STRUCT(Aperture::particle_data, x1, dx1,
                           p1, gamma, cell, flag);

BOOST_FUSION_ADAPT_STRUCT(Aperture::photon_data, x1, p1, path_left, cell, flag);

#endif  // _PARTICLE_DATA_H_
