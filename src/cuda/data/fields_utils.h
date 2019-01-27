#ifndef _FIELDS_UTILS_H_
#define _FIELDS_UTILS_H_

#include "core/typedefs.h"

namespace Aperture {

template <typename T>
class cu_vector_field;
template <typename T>
class cu_scalar_field;

/// Compute v = v + q * u, where q is a constant
void field_add(cu_vector_field<Scalar>& v, const cu_vector_field<Scalar>& u,
               Scalar q);

/// Compute v = v + q * u, where q is a scalar field
void field_add(cu_vector_field<Scalar>& v, const cu_vector_field<Scalar>& u,
               const cu_scalar_field<Scalar>& q);

}  // namespace Aperture

#endif  // _FIELDS_UTILS_H_
