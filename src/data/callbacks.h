#ifndef  _CALLBACKS_H_
#define  _CALLBACKS_H_

#include <functional>
#include "data/fields.h"
#include "data/particles.h"
#include "data/photons.h"

namespace Aperture {

typedef std::function<void(VectorField<Scalar>&)> vfield_comm_callback;

typedef std::function<void(ScalarField<Scalar>&)> sfield_comm_callback;

typedef std::function<void(VectorField<Scalar>&, VectorField<Scalar>&, double)> fieldBC_callback;

typedef std::function<void(Particles&)> ptc_comm_callback;
typedef std::function<void(Photons&)> photon_comm_callback;

}

#endif   // _CALLBACKS_H_
