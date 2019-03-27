#ifndef _ITERATE_DEVICES_H_
#define _ITERATE_DEVICES_H_

#include "cuda/cudaUtility.h"
#include "utils/logger.h"
#include <vector>

namespace Aperture {

template <typename Func>
void
for_each_device(const std::vector<int> &dev_map, const Func &f) {
  for (int n = 0; n < dev_map.size(); n++) {
    int dev_id = dev_map[n];
    CudaSafeCall(cudaSetDevice(dev_id));
    // Logger::print_debug("using device {}", dev_id);
    f(n);
    // CudaSafeCall(cudaDeviceSynchronize());
  }
}

} // namespace Aperture

#endif // _ITERATE_DEVICES_H_
