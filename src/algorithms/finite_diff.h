#ifndef _FINITE_DIFF_H_
#define _FINITE_DIFF_H_

#include "data/fields_dev.h"

namespace Aperture {

void curl_add(cu_vector_field<Scalar>& result, const cu_vector_field<Scalar>& u,
              Scalar q = 1.0);
void curl(cu_vector_field<Scalar>& result, const cu_vector_field<Scalar>& u,
          Scalar q = 1.0);
void div_add(cu_scalar_field<Scalar>& result, const cu_vector_field<Scalar>& u,
             Scalar q = 1.0);
void div(cu_scalar_field<Scalar>& result, const cu_vector_field<Scalar>& u,
         Scalar q = 1.0);
void grad_add(cu_vector_field<Scalar>& result, const cu_scalar_field<Scalar>& f,
              Scalar q = 1.0);
void grad(cu_vector_field<Scalar>& result, const cu_scalar_field<Scalar>& f,
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
