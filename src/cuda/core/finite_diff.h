#ifndef _FINITE_DIFF_H_
#define _FINITE_DIFF_H_

#include "core/fields.h"

namespace Aperture {

void curl_add(vector_field<Scalar>& result, vector_field<Scalar>& u,
              Scalar q = 1.0);
void curl(vector_field<Scalar>& result, vector_field<Scalar>& u,
          Scalar q = 1.0);
void div_add(scalar_field<Scalar>& result, vector_field<Scalar>& u,
             Scalar q = 1.0);
void div(scalar_field<Scalar>& result, vector_field<Scalar>& u,
         Scalar q = 1.0);
void grad_add(vector_field<Scalar>& result, scalar_field<Scalar>& f,
              Scalar q = 1.0);
void grad(vector_field<Scalar>& result, scalar_field<Scalar>& f,
          Scalar q = 1.0);
// void ffe_edotb(cu_scalar_field<Scalar>& result, const
// cu_vector_field<Scalar>& E,
//                const cu_vector_field<Scalar>& B, Scalar q = 1.0);
// void ffe_j(cu_vector_field<Scalar>& result, const cu_scalar_field<Scalar>&
// tmp_f,
//            const cu_vector_field<Scalar>& E, const cu_vector_field<Scalar>&
//            B, Scalar q = 1.0);
}  // namespace Aperture

#endif  // _FINITE_DIFF_H_
