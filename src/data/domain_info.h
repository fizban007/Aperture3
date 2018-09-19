#ifndef _DOMAIN_INFO_H_
#define _DOMAIN_INFO_H_

#include "constant_defs.h"
#include "data/enum_types.h"
#include "data/vec3.h"
#include <array>
#include <vector>

namespace Aperture {

struct DomainInfo {
 public:
  int dim = 1;
  int rank = 0;
  std::array<bool, 6> is_boundary = {false, false, false,
                                     false, false, false};
  std::array<bool, 3> is_periodic = {
      false, false, false};  ///< Marks whether the domain is
                             ///  periodic in each direction
  std::array<int, 3>
      cart_neighbor_right;  ///< Ranks of the right neighbors in each
                            ///< direction
  std::array<int, 3>
      cart_neighbor_left;  ///< Ranks of the left neighbors in each
                           ///< direction
  std::array<int, 3> cart_dims;
  Index cart_pos;

  // std::vector<std::vector<std::vector<int>>>
  //     rank_map;  ///< Rank map of the domain decomposition
  ProcessState state = ProcessState::idle;
};  // ----- end of class domain_info -----
}  // namespace Aperture

#endif  // _DOMAIN_INFO_H_
