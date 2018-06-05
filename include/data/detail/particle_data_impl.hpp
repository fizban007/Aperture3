#ifndef _PARTICLE_DATA_IMPL_H_
#define _PARTICLE_DATA_IMPL_H_

namespace Aperture {

struct read_at_idx
{
  size_t idx_;
  HOST_DEVICE read_at_idx(size_t idx) : idx_(idx) {}

  template <typename T>
  HD_INLINE void operator()(const T& x) const {
    boost::fusion::at_c<0>(x) = boost::fusion::at_c<1>(x)[idx_];
  }
};

single_particle_t particle_data::operator[](size_t idx) const {
  single_particle_t part;
  typedef boost::fusion::vector<single_particle_t&, const particle_data&> seq;
  boost::fusion::for_each(boost::fusion::zip_view<seq>(seq(part, *this)),
                          read_at_idx(idx) );
  return part;
}

single_photon_t photon_data::operator[](size_t idx) const {
  single_photon_t part;
  typedef boost::fusion::vector<single_photon_t&, const photon_data&> seq;
  boost::fusion::for_each(boost::fusion::zip_view<seq>(seq(part, *this)),
                          read_at_idx(idx) );
  return part;
}

}


#endif  // _PARTICLE_DATA_IMPL_H_
