#ifndef _DOMAIN_INFO_H_
#define _DOMAIN_INFO_H_

#include "core/constant_defs.h"
#include "core/enum_types.h"
#include "core/vec3.h"
#include <array>
#include <vector>

namespace Aperture {

struct domain_info {
 public:
  int size = 1;
  int rank = 0;
  bool is_boundary[6] = {false, false, false, false, false, false};
  int is_periodic[3] = {false, false,
                         false};  ///< Marks whether the domain is
                                  ///  periodic in each direction
  int neighbor_right[3];  ///< Ranks of the right neighbors in each
                          ///< direction
  int neighbor_left[3];   ///< Ranks of the left neighbors in each
                          ///< direction
  int cart_dims[3] = {1, 1, 1};
  int cart_coord[3] = {0, 0, 0};

  // std::vector<std::vector<std::vector<int>>>
  //     rank_map;  ///< Rank map of the domain decomposition
  // ProcessState state = ProcessState::idle;
};  // ----- end of class domain_info -----
}  // namespace Aperture

#endif  // _DOMAIN_INFO_H_
