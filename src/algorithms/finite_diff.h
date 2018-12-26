#ifndef _FINITE_DIFF_H_
#define _FINITE_DIFF_H_

#include "data/fields_dev.h"

namespace Aperture {

void curl_add(VectorField<Scalar>& result, const VectorField<Scalar>& u,
              Scalar q = 1.0);
void curl(VectorField<Scalar>& result, const VectorField<Scalar>& u,
          Scalar q = 1.0);
void div_add(ScalarField<Scalar>& result, const VectorField<Scalar>& u,
             Scalar q = 1.0);
void div(ScalarField<Scalar>& result, const VectorField<Scalar>& u,
         Scalar q = 1.0);
void grad_add(VectorField<Scalar>& result, const ScalarField<Scalar>& f,
              Scalar q = 1.0);
void grad(VectorField<Scalar>& result, const ScalarField<Scalar>& f,
          Scalar q = 1.0);
// void ffe_edotb(ScalarField<Scalar>& result, const
// VectorField<Scalar>& E,
//                const VectorField<Scalar>& B, Scalar q = 1.0);
// void ffe_j(VectorField<Scalar>& result, const ScalarField<Scalar>&
// tmp_f,
//            const VectorField<Scalar>& E, const VectorField<Scalar>&
//            B, Scalar q = 1.0);
}  // namespace Aperture

#endif  // _FINITE_DIFF_H_
