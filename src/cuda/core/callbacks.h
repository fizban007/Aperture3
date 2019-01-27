#ifndef _CALLBACKS_H_
#define _CALLBACKS_H_

#include "cuda/data/fields_dev.h"
#include "cuda/data/particles_dev.h"
#include "cuda/data/photons_dev.h"
#include <functional>

namespace Aperture {

typedef std::function<void(cu_vector_field<Scalar>&)> vfield_comm_callback;

typedef std::function<void(cu_scalar_field<Scalar>&)> sfield_comm_callback;

typedef std::function<void(cu_vector_field<Scalar>&, cu_vector_field<Scalar>&,
                           double)>
    fieldBC_callback;

typedef std::function<void(Particles&)> ptc_comm_callback;
typedef std::function<void(Photons&)> photon_comm_callback;

}  // namespace Aperture

#endif  // _CALLBACKS_H_
