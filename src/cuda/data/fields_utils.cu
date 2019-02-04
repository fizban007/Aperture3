#include "core/constant_defs.h"
#include "core/detail/multi_array_utils.hpp"
#include "cuda/cudaUtility.h"
#include "cuda/data/detail/multi_array_utils.cuh"
#include "cuda/data/fields_dev.h"
#include "cuda/data/fields_utils.h"

namespace Aperture {

void
field_add(cu_vector_field<Scalar>& v, const cu_vector_field<Scalar>& u,
          Scalar q) {
  auto& mesh = v.grid().mesh();
  dim3 blockSize(16, 8, 8);
  dim3 gridSize(16, 8, 8);
  for (int i = 0; i < VECTOR_DIM; i++) {
    Kernels::map_array_binary_op<Scalar>
        <<<gridSize, blockSize>>>(v.ptr(i), u.ptr(i), mesh.extent(),
                                  detail::Op_MultConstAdd<Scalar>{q});
    CudaCheckError();
  }
}

void
field_add(cu_vector_field<Scalar>& v, const cu_vector_field<Scalar>& u,
          const cu_scalar_field<Scalar>& q) {}

}  // namespace Aperture
