#ifndef _FIELD_SOLVER_HELPER_H_
#define _FIELD_SOLVER_HELPER_H_


namespace Aperture {

namespace Kernels {

template <int Order> struct Pad;

template <>
struct Pad<2> { enum { val = 1 }; };

template <>
struct Pad<4> { enum { val = 2 }; };

template <>
struct Pad<6> { enum { val = 3 }; };

template <int Order, int DIM1, int DIM2, int DIM3>
__device__ __forceinline__
void init_shared_memory(Scalar s_u1[][DIM2 + Pad<Order>::val*2][DIM1 + Pad<Order>::val*2],
                        Scalar s_u2[][DIM2 + Pad<Order>::val*2][DIM1 + Pad<Order>::val*2],
                        Scalar s_u3[][DIM2 + Pad<Order>::val*2][DIM1 + Pad<Order>::val*2],
                        cudaPitchedPtr& u1, cudaPitchedPtr& u2, cudaPitchedPtr& u3,
                        size_t globalOffset, int c1, int c2, int c3) {
  // Load field values into shared memory
  s_u1[c3][c2][c1] = *(Scalar*)((char*)u1.ptr + globalOffset);
  s_u2[c3][c2][c1] = *(Scalar*)((char*)u2.ptr + globalOffset);
  s_u3[c3][c2][c1] = *(Scalar*)((char*)u3.ptr + globalOffset);

  // Handle extra guard cells
  if (c1 < 2*Pad<Order>::val) {
    s_u1[c3][c2][c1 - Pad<Order>::val] =
        *(Scalar*)((char*)u1.ptr + globalOffset - Pad<Order>::val*sizeof(Scalar));
    s_u2[c3][c2][c1 - Pad<Order>::val] =
        *(Scalar*)((char*)u2.ptr + globalOffset - Pad<Order>::val*sizeof(Scalar));
    s_u3[c3][c2][c1 - Pad<Order>::val] =
        *(Scalar*)((char*)u3.ptr + globalOffset - Pad<Order>::val*sizeof(Scalar));
    s_u1[c3][c2][c1 + DIM1] =
        *(Scalar*)((char*)u1.ptr + globalOffset + DIM1*sizeof(Scalar));
    s_u2[c3][c2][c1 + DIM1] =
        *(Scalar*)((char*)u2.ptr + globalOffset + DIM1*sizeof(Scalar));
    s_u3[c3][c2][c1 + DIM1] =
        *(Scalar*)((char*)u3.ptr + globalOffset + DIM1*sizeof(Scalar));
  }
  if (c2 < 2*Pad<Order>::val) {
    s_u1[c3][c2 - Pad<Order>::val][c1] =
        *(Scalar*)((char*)u1.ptr + globalOffset - Pad<Order>::val*u1.pitch);
    s_u2[c3][c2 - Pad<Order>::val][c1] =
        *(Scalar*)((char*)u2.ptr + globalOffset - Pad<Order>::val*u2.pitch);
    s_u3[c3][c2 - Pad<Order>::val][c1] =
        *(Scalar*)((char*)u3.ptr + globalOffset - Pad<Order>::val*u3.pitch);
    s_u1[c3][c2 + DIM2][c1] =
        *(Scalar*)((char*)u1.ptr + globalOffset + DIM2*u1.pitch);
    s_u2[c3][c2 + DIM2][c1] =
        *(Scalar*)((char*)u2.ptr + globalOffset + DIM2*u2.pitch);
    s_u3[c3][c2 + DIM2][c1] =
        *(Scalar*)((char*)u3.ptr + globalOffset + DIM2*u3.pitch);
  }
  if (c3 < 2*Pad<Order>::val) {
    s_u1[c3 - Pad<Order>::val][c2][c1] =
        *(Scalar*)((char*)u1.ptr + globalOffset - Pad<Order>::val*u1.pitch * u1.ysize);
    s_u2[c3 - Pad<Order>::val][c2][c1] =
        *(Scalar*)((char*)u2.ptr + globalOffset - Pad<Order>::val*u2.pitch * u2.ysize);
    s_u3[c3 - Pad<Order>::val][c2][c1] =
        *(Scalar*)((char*)u3.ptr + globalOffset - Pad<Order>::val*u3.pitch * u3.ysize);
    s_u1[c3 + DIM3][c2][c1] =
        *(Scalar*)((char*)u1.ptr + globalOffset + DIM3*u1.pitch * u1.ysize);
    s_u2[c3 + DIM3][c2][c1] =
        *(Scalar*)((char*)u2.ptr + globalOffset + DIM3*u2.pitch * u2.ysize);
    s_u3[c3 + DIM3][c2][c1] =
        *(Scalar*)((char*)u3.ptr + globalOffset + DIM3*u3.pitch * u3.ysize);
  }
}

template <int Order, int DIM1, int DIM2, int DIM3>
__device__ __forceinline__
void init_shared_memory(Scalar s_f[][DIM2 + Pad<Order>::val*2][DIM1 + Pad<Order>::val*2],
                        cudaPitchedPtr& f, size_t globalOffset, int c1, int c2, int c3) {
  // Load field values into shared memory
  s_f[c3][c2][c1] = *(Scalar*)((char*)f.ptr + globalOffset);

  // Handle extra guard cells
  if (c1 < 2*Pad<Order>::val) {
    s_f[c3][c2][c1 - Pad<Order>::val] =
        *(Scalar*)((char*)f.ptr + globalOffset - Pad<Order>::val*sizeof(Scalar));
    s_f[c3][c2][c1 + DIM1] =
        *(Scalar*)((char*)f.ptr + globalOffset + DIM1*sizeof(Scalar));
  }
  if (c2 < 2*Pad<Order>::val) {
    s_f[c3][c2 - Pad<Order>::val][c1] =
        *(Scalar*)((char*)f.ptr + globalOffset - Pad<Order>::val*f.pitch);
    s_f[c3][c2 + DIM2][c1] =
        *(Scalar*)((char*)f.ptr + globalOffset + DIM2*f.pitch);
  }
  if (c3 < 2*Pad<Order>::val) {
    s_f[c3 - Pad<Order>::val][c2][c1] =
        *(Scalar*)((char*)f.ptr + globalOffset - Pad<Order>::val*f.pitch * f.ysize);
    s_f[c3 + DIM3][c2][c1] =
        *(Scalar*)((char*)f.ptr + globalOffset + DIM3*f.pitch * f.ysize);
  }
}



}

}


#endif  // _FIELD_SOLVER_HELPER_H_
