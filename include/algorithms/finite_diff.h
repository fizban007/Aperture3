#ifndef _FINITE_DIFF_H_
#define _FINITE_DIFF_H_

#include "data/fields.h"

namespace Aperture {

void curl(VectorField<Scalar>& result, const VectorField<Scalar>& u);
void curl_2(VectorField<Scalar>& result, const VectorField<Scalar>& u);
void div(ScalarField<Scalar>& result, const VectorField<Scalar>& u);
void grad(VectorField<Scalar>& result, const ScalarField<Scalar>& f);
void grad_2(VectorField<Scalar>& result, const ScalarField<Scalar>& f);

}

#endif  // _FINITE_DIFF_H_
