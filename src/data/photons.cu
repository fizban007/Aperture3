#include "data/photons.h"
#include "sim_environment.h"
#include "utils/logger.h"
#include "utils/util_functions.h"
#include "algorithms/functions.h"
#include "cuda/cuda_control.h"
#include "cuda/cudaUtility.h"

namespace Aperture {

Photons::Photons() {}

Photons::Photons(std::size_t max_num)
    : ParticleBase<single_photon_t>(max_num) {}

Photons::Photons(const Environment& env)
    : ParticleBase<single_photon_t>(env.params().max_photon_number) {
}

Photons::~Photons() {}

}
