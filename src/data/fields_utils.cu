#include "constant_defs.h"
#include "cuda/cudaUtility.h"
#include "data/detail/multi_array_utils.hpp"
#include "data/fields_dev.h"
#include "data/fields_utils.h"

namespace Aperture {

void
field_add(VectorField<Scalar>& v, const VectorField<Scalar>& u,
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
field_add(VectorField<Scalar>& v, const VectorField<Scalar>& u,
          const ScalarField<Scalar>& q) {}

}  // namespace Aperture
