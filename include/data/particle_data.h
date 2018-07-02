#ifndef _PARTICLE_DATA_H_
#define _PARTICLE_DATA_H_

#include "constant_defs.h"
#include "data/typedefs.h"
#include "data/vec3.h"
#include "cuda/cuda_control.h"
#include "visit_struct/visit_struct.hpp"
#include "data/detail/macro_trickery.h"
#include <cinttypes>
#include <type_traits>
#include <cinttypes>

namespace Aperture {
  template <typename SingleType>
  struct particle_array_type;
}

// Here we define particle types through heavy macro magic. The macro
// DEF_PARTICLE_STRUCT takes in a name, and a sequence of triples, each one
// defines an entry in the particle struct. The macro defines two structs at the
// same time: `particle_data` and `single_particle_t`. For example, the following
// definition:
//
//     DEF_PARTICLE_STRUCT(particle,
//                         (float, x1, 0.0)
//                         (float, x2, 0.0)
//                         (float, x3, 0.0));
//
// will define two structs:
//
//     struct single_particle_t {
//       float x1 = 0.0;
//       float x2 = 0.0;
//       float x3 = 0.0;
//     };
//     struct particle_data {
//       float* x1;
//       float* x2;
//       float* x3;
//       enum { size = 3 * sizeof(float) };
//       single_particle_t operator[](size_t idx) const;
//     };
//
// where `single_particle_t` is a struct representing a single particle, and
// `particle_data` is a struct of arrays that contain the actual data. An
// integral constant `size` is defined in `particle_data` for MPI purposes, and
// an index operator is defined to easily read a single particle.

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

#endif  // _PARTICLE_DATA_H_
