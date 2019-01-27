#ifndef _GRID_LOG_SPH_H_
#define _GRID_LOG_SPH_H_

#include "grids/grid_log_sph_base.h"
#include "data/fields.h"

namespace Aperture {

class Grid_LogSph : public Grid_LogSph_base<Grid_LogSph> {
 public:
  typedef Grid_LogSph_base<Grid_LogSph> base_class;

  Grid_LogSph();
  virtual ~Grid_LogSph();

  virtual void init(const SimParams& params) override;

  void compute_flux(scalar_field<Scalar>& flux,
                    vector_field<Scalar>& B,
                    vector_field<Scalar>& B_bg) const;

};  // ----- end of class Grid_LogSph : public
    // Grid_LogSph_base<Grid_LogSph> -----

}  // namespace Aperture

#endif  // _GRID_LOG_SPH_H_
