#include "H5Cpp.h"
#include "cu_exporter.h"
#include "cuda/constant_mem.h"
#include "cuda/core/cu_sim_data.h"
#include "cuda/core/cu_sim_environment.h"
#include "cuda/cudaUtility.h"
#include <vector>

using namespace H5;

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
cu_exporter::write_output(cu_sim_data& data, uint32_t timestep,
                          double time) {
  H5File datafile(fmt::format("{}{}{:06d}.h5", outputDirectory,
                              filePrefix, timestep)
                      .c_str(),
                  H5F_ACC_RDWR | H5F_ACC_TRUNC);
  add_grid_output(
      data, "E1",
      [] __host__ __device__(cu_sim_data::data_ptrs & ptrs,
                             pitchptr<Scalar> & p, Index idx,
                             Index idx_out) {
        p(idx_out) = ptrs.E1(idx) + ptrs.Ebg1(idx);
      },
      datafile);
  add_grid_output(
      data, "E2",
      [] __host__ __device__(cu_sim_data::data_ptrs & ptrs,
                             pitchptr<Scalar> & p, Index idx,
                             Index idx_out) {
        p(idx_out) = ptrs.E2(idx) + ptrs.Ebg2(idx);
      },
      datafile);
  add_grid_output(
      data, "E3",
      [] __host__ __device__(cu_sim_data::data_ptrs & ptrs,
                             pitchptr<Scalar> & p, Index idx,
                             Index idx_out) {
        p(idx_out) = ptrs.E3(idx) + ptrs.Ebg3(idx);
      },
      datafile);
  add_grid_output(
      data, "B1",
      [] __host__ __device__(cu_sim_data::data_ptrs & ptrs,
                             pitchptr<Scalar> & p, Index idx,
                             Index idx_out) {
        p(idx_out) = ptrs.B1(idx) + ptrs.Bbg1(idx);
      },
      datafile);
  add_grid_output(
      data, "B2",
      [] __host__ __device__(cu_sim_data::data_ptrs & ptrs,
                             pitchptr<Scalar> & p, Index idx,
                             Index idx_out) {
        p(idx_out) = ptrs.B2(idx) + ptrs.Bbg2(idx);
      },
      datafile);
  add_grid_output(
      data, "B3",
      [] __host__ __device__(cu_sim_data::data_ptrs & ptrs,
                             pitchptr<Scalar> & p, Index idx,
                             Index idx_out) {
        p(idx_out) = ptrs.B3(idx) + ptrs.Bbg3(idx);
      },
      datafile);
  add_grid_output(
      data, "J1",
      [] __host__ __device__(
          cu_sim_data::data_ptrs & ptrs, pitchptr<Scalar> & p,
          Index idx, Index idx_out) { p(idx_out) = ptrs.J1(idx); },
      datafile);
  add_grid_output(
      data, "J2",
      [] __host__ __device__(
          cu_sim_data::data_ptrs & ptrs, pitchptr<Scalar> & p,
          Index idx, Index idx_out) { p(idx_out) = ptrs.J2(idx); },
      datafile);
  add_grid_output(
      data, "J3",
      [] __host__ __device__(
          cu_sim_data::data_ptrs & ptrs, pitchptr<Scalar> & p,
          Index idx, Index idx_out) { p(idx_out) = ptrs.J3(idx); },
      datafile);
  add_grid_output(
      data, "Rho_e",
      [] __host__ __device__(
          cu_sim_data::data_ptrs & ptrs, pitchptr<Scalar> & p,
          Index idx, Index idx_out) { p(idx_out) = ptrs.Rho[0](idx); },
      datafile);
  add_grid_output(
      data, "Rho_p",
      [] __host__ __device__(
          cu_sim_data::data_ptrs & ptrs, pitchptr<Scalar> & p,
          Index idx, Index idx_out) { p(idx_out) = ptrs.Rho[1](idx); },
      datafile);
  if (data.env.params().num_species > 2) {
    add_grid_output(
        data, "Rho_i",
        [] __host__ __device__(cu_sim_data::data_ptrs & ptrs,
                               pitchptr<Scalar> & p, Index idx,
                               Index idx_out) {
          p(idx_out) = ptrs.Rho[2](idx);
        },
        datafile);
  }
  add_grid_output(
      data, "photon_produced",
      [] __host__ __device__(cu_sim_data::data_ptrs & ptrs,
                             pitchptr<Scalar> & p, Index idx,
                             Index idx_out) {
        p(idx_out) = ptrs.photon_produced(idx);
      },
      datafile);
  add_grid_output(
      data, "pair_produced",
      [] __host__ __device__(cu_sim_data::data_ptrs & ptrs,
                             pitchptr<Scalar> & p, Index idx,
                             Index idx_out) {
        p(idx_out) = ptrs.pair_produced(idx);
      },
      datafile);
  add_grid_output(
      data, "photon_num",
      [] __host__ __device__(cu_sim_data::data_ptrs & ptrs,
                             pitchptr<Scalar> & p, Index idx,
                             Index idx_out) {
        p(idx_out) = ptrs.photon_num(idx);
      },
      datafile);
  add_grid_output(
      data, "divE",
      [] __host__ __device__(
          cu_sim_data::data_ptrs & ptrs, pitchptr<Scalar> & p,
          Index idx, Index idx_out) { p(idx_out) = ptrs.divE(idx); },
      datafile);
  add_grid_output(
      data, "divB",
      [] __host__ __device__(
          cu_sim_data::data_ptrs & ptrs, pitchptr<Scalar> & p,
          Index idx, Index idx_out) { p(idx_out) = ptrs.divB(idx); },
      datafile);
  add_grid_output(
      data, "EdotB_avg",
      [] __host__ __device__(
          cu_sim_data::data_ptrs & ptrs, pitchptr<Scalar> & p,
          Index idx, Index idx_out) { p(idx_out) = ptrs.EdotB(idx); },
      datafile);
  add_grid_output(
      data, "EdotB",
      [] __host__ __device__(cu_sim_data::data_ptrs & ptrs,
                             pitchptr<Scalar> & p, Index idx,
                             Index idx_out) {
        p(idx_out) = ptrs.E1(idx) * ptrs.B1(idx) +
                     ptrs.E2(idx) * ptrs.B2(idx) +
                     ptrs.E3(idx) * ptrs.B3(idx);
      },
      datafile);

  datafile.close();
}

template <typename Func>
void
cu_exporter::add_grid_output(cu_sim_data& data, const std::string& name,
                             Func f, H5File& file) {
  if (data.env.grid().dim() == 2) {
    dim3 grid_size(32, 32);
    dim3 block_size(32, 32);
    Kernels::sample_grid_quantity2d<<<grid_size, block_size>>>(
        data.get_ptrs(), tmp_grid_cudata.extent(),
        m_env.params().downsample, tmp_grid_cudata.data_d(), f);
    CudaCheckError();

    tmp_grid_cudata.sync_to_host();

    // Actually write the temp array to hdf
    hsize_t dims[2] = {(uint32_t)tmp_grid_cudata.width(),
                       (uint32_t)tmp_grid_cudata.height()};
    DataSpace dataspace(2, dims);
    DataSet dataset =
        file.createDataSet(name, PredType::NATIVE_FLOAT, dataspace);
    dataset.write(tmp_grid_cudata.data(), PredType::NATIVE_FLOAT);
  }
}

}  // namespace Aperture
