#ifndef _CALLBACKS_H_
#define _CALLBACKS_H_

#include "data/fields_dev.h"
#include "data/particles_dev.h"
#include "data/photons.h"
#include <functional>

namespace Aperture {

typedef std::function<void(VectorField<Scalar>&)> vfield_comm_callback;

typedef std::function<void(ScalarField<Scalar>&)> sfield_comm_callback;

typedef std::function<void(VectorField<Scalar>&, VectorField<Scalar>&,
                           double)>
    fieldBC_callback;

typedef std::function<void(Particles&)> ptc_comm_callback;
typedef std::function<void(Photons&)> photon_comm_callback;

}  // namespace Aperture

#endif  // _CALLBACKS_H_
