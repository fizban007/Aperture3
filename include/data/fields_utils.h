#ifndef _FIELDS_UTILS_H_
#define _FIELDS_UTILS_H_

#include "data/typedefs.h"

namespace Aperture {

template <typename T>
class VectorField;
template <typename T>
class ScalarField;

/// Compute v = v + q * u, where q is a constant
void field_add(VectorField<Scalar>& v, const VectorField<Scalar>& u, Scalar q);

/// Compute v = v + q * u, where q is a scalar field
void field_add(VectorField<Scalar>& v, const VectorField<Scalar>& u,
               const ScalarField<Scalar>& q);

}

#endif  // _FIELDS_UTILS_H_
