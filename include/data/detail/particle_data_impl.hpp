#ifndef _PARTICLE_DATA_IMPL_H_
#define _PARTICLE_DATA_IMPL_H_

namespace Aperture {

single_particle_t particle_data::operator[](int idx) const {
  single_particle_t part;
  typedef boost::fusion::vector<single_particle_t&, const particle_data&> seq;
  boost::fusion::for_each(boost::fusion::zip_view<seq>(seq(part, *this)),
                          [this, idx](const auto& x) {
                            boost::fusion::at_c<0>(x) =
                                boost::fusion::at_c<1>(x)[idx];
                          });
  return part;
}

single_photon_t photon_data::operator[](int idx) const {
  single_photon_t part;
  typedef boost::fusion::vector<single_photon_t&, const photon_data&> seq;
  boost::fusion::for_each(boost::fusion::zip_view<seq>(seq(part, *this)),
                          [this, idx](const auto& x) {
                            boost::fusion::at_c<0>(x) =
                                boost::fusion::at_c<1>(x)[idx];
                          });
  return part;
}

}


#endif  // _PARTICLE_DATA_IMPL_H_
