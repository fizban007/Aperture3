#include "cu_exporter.h"
#include "cuda/constant_mem.h"
#include "cuda/core/cu_sim_data.h"
#include "cuda/core/cu_sim_environment.h"
#include "cuda/cudaUtility.h"

namespace Aperture {

namespace Kernels {

template <typename Func>
__global__ void
sample_grid_quantity2d(cu_sim_data::data_ptrs ptrs, Extent ext,
                       int downsample, pitchptr<Scalar> result,
                       Func f) {
  for (int j = threadIdx.y + blockIdx.y * blockDim.y; j < ext.height();
       j += gridDim.y * blockDim.y) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < ext.width();
         i += gridDim.x * blockDim.x) {
      Index idx_out(i, j, 0);
      Index idx_data(i * downsample + dev_mesh.guard[0],
                     j * downsample + dev_mesh.guard[1], 0);
      f(ptrs, result, idx_data, idx_out);
    }
  }
}

}  // namespace Kernels

cu_exporter::cu_exporter(cu_sim_environment& env, uint32_t& timestep)
    : exporter(env, timestep) {
  auto& mesh = m_env.local_grid().mesh();
  auto ext = mesh.extent_less();
  auto d = m_env.params().downsample;
  tmp_grid_cudata.resize(ext.width() / d, ext.height() / d,
                         ext.depth() / d);
}

cu_exporter::~cu_exporter() {}

void
cu_exporter::write_output(cu_sim_data& data) {
  add_grid_output(data, "E1",
                  [] __host__ __device__(cu_sim_data::data_ptrs & ptrs,
                                         pitchptr<Scalar> & p,
                                         Index idx, Index idx_out) {
                    p(idx_out) = ptrs.E1(idx) + ptrs.Ebg1(idx);
                  });
  add_grid_output(data, "E2",
                  [] __host__ __device__(cu_sim_data::data_ptrs & ptrs,
                                         pitchptr<Scalar> & p,
                                         Index idx, Index idx_out) {
                    p(idx_out) = ptrs.E2(idx) + ptrs.Ebg2(idx);
                  });
  add_grid_output(data, "E3",
                  [] __host__ __device__(cu_sim_data::data_ptrs & ptrs,
                                         pitchptr<Scalar> & p,
                                         Index idx, Index idx_out) {
                    p(idx_out) = ptrs.E3(idx) + ptrs.Ebg3(idx);
                  });
  add_grid_output(data, "B1",
                  [] __host__ __device__(cu_sim_data::data_ptrs & ptrs,
                                         pitchptr<Scalar> & p,
                                         Index idx, Index idx_out) {
                    p(idx_out) = ptrs.B1(idx) + ptrs.Bbg1(idx);
                  });
  add_grid_output(data, "B2",
                  [] __host__ __device__(cu_sim_data::data_ptrs & ptrs,
                                         pitchptr<Scalar> & p,
                                         Index idx, Index idx_out) {
                    p(idx_out) = ptrs.B2(idx) + ptrs.Bbg2(idx);
                  });
  add_grid_output(data, "B3",
                  [] __host__ __device__(cu_sim_data::data_ptrs & ptrs,
                                         pitchptr<Scalar> & p,
                                         Index idx, Index idx_out) {
                    p(idx_out) = ptrs.B3(idx) + ptrs.Bbg3(idx);
                  });
  add_grid_output(data, "J1",
                  [] __host__ __device__(cu_sim_data::data_ptrs & ptrs,
                                         pitchptr<Scalar> & p,
                                         Index idx, Index idx_out) {
                    p(idx_out) = ptrs.J1(idx);
                  });
  add_grid_output(data, "J2",
                  [] __host__ __device__(cu_sim_data::data_ptrs & ptrs,
                                         pitchptr<Scalar> & p,
                                         Index idx, Index idx_out) {
                    p(idx_out) = ptrs.J2(idx);
                  });
  add_grid_output(data, "J3",
                  [] __host__ __device__(cu_sim_data::data_ptrs & ptrs,
                                         pitchptr<Scalar> & p,
                                         Index idx, Index idx_out) {
                    p(idx_out) = ptrs.J3(idx);
                  });
}

template <typename Func>
void
cu_exporter::add_grid_output(cu_sim_data& data, const std::string& name,
                             Func f) {
  if (data.env.grid().dim() == 2) {
    dim3 grid_size(32, 32);
    dim3 block_size(32, 32);
    Kernels::sample_grid_quantity2d<<<grid_size, block_size>>>(
        data.get_ptrs(), tmp_grid_cudata.extent(),
        m_env.params().downsample, tmp_grid_cudata.data_d(), f);
    CudaCheckError();

    tmp_grid_cudata.sync_to_host();
  }
}

}  // namespace Aperture
