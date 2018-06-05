#include "data/particles.h"
#include <cuda.h>
#include "catch.hpp"

using namespace Aperture;

__global__
void set_cells(uint32_t* cells) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  cells[idx] = idx;
}

TEST_CASE("Initializing and Adding particles", "[Particles]") {
  size_t N = 100000;
  Particles ptc(N);

  CHECK(ptc.type() == ParticleType::electron);
  CHECK(ptc.charge() == -1.0);
  CHECK(ptc.mass() == 1.0);
  CHECK(ptc.number() == 0);
  CHECK(ptc.numMax() == N);

  ptc.append(0.5, 1.0, 100);
  ptc.append(0.2, 1.0, 200);
  ptc.append(0.2, 0.0, 300);
  ptc.append(0.2, -1.0, 400);

  CHECK(ptc.number() == 4);
  CHECK(ptc.data().x1[0] == Approx(0.5));
  CHECK(ptc.data().p1[3] == -1.0);
  CHECK(ptc.data().cell[2] == 300);

  set_cells<<<256, 256>>>(ptc.data().cell);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  for (size_t i = 0; i < 256*256; i++) {
    CHECK(ptc.data().cell[i] == i);
  }
}
