#ifndef _FIELD_DATA_H_
#define _FIELD_DATA_H_

#include <cuda_runtime.h>

namespace Aperture {

struct FieldData
{
  cudaPitchedPtr E1, E2, E3;
  cudaPitchedPtr B1, B2, B3;
};

}

#endif  // _FIELD_DATA_H_
