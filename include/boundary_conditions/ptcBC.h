#ifndef _PTCBC_H_
#define _PTCBC_H_

namespace Aperture {

class ptcBC
{
 public:
  ptcBC() {}
  virtual ~ptcBC() {}

  // virtual void apply (PICData& data, double time) = 0;
}; // ----- end of class ptcBC -----

}

#endif  // _PTCBC_H_
