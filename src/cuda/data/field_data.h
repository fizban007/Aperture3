#ifndef _FIELD_DATA_H_
#define _FIELD_DATA_H_

#include "cuda/utils/pitchptr.cuh"
#include "core/typedefs.h"

namespace Aperture {

struct FieldData
{
  pitchptr<Scalar> E1, E2, E3;
  pitchptr<Scalar> B1, B2, B3;
};

}

#endif  // _FIELD_DATA_H_
