#ifndef _FIELD_DATA_H_
#define _FIELD_DATA_H_

#include "cuda/utils/typed_pitchedptr.cuh"
#include "core/typedefs.h"

namespace Aperture {

struct FieldData
{
  typed_pitchedptr<Scalar> E1, E2, E3;
  typed_pitchedptr<Scalar> B1, B2, B3;
};

}

#endif  // _FIELD_DATA_H_
