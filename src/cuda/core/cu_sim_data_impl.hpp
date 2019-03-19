#ifndef _CU_SIM_DATA_IMPL_H_
#define _CU_SIM_DATA_IMPL_H_

#include "cu_sim_data.h"
#include "cuda/cudaUtility.h"

namespace Aperture {

template <class Func>
void cu_sim_data::init_bg_B_field(int component, const Func &f) {
  for (int n = 0; n < dev_map.size(); n++) {
    int dev_id = dev_map[n];
    CudaSafeCall(cudaSetDevice(dev_id));

    Bbg[n].initialize(component, f);
    Bbg[n].sync_to_device();
  }
}

template <class Func>
void cu_sim_data::init_bg_E_field(int component, const Func &f) {
  for (int n = 0; n < dev_map.size(); n++) {
    int dev_id = dev_map[n];
    CudaSafeCall(cudaSetDevice(dev_id));

    Ebg[n].initialize(component, f);
    Ebg[n].sync_to_device();
  }
}

}


#endif  // _CU_SIM_DATA_IMPL_H_
