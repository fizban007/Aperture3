#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

#include <limits>
#include <math.h>

#define VECTOR_DIM 3
#define CONST_PI 3.14159265358979323846
// #define MAX_CELL std::numeric_limits<uint32_t>::max()
// Note: you can only have up to 4 billion cells on one rank!
#define MAX_CELL 4294967295
#define MAX_TILE 4294967295
#define EPS 1.0e-10

#define CENTRAL_ZONE 13
#define ZONE_NUMBER 27

#define NUM_BOUNDARIES 6
// Number of particle communication buffers
#define NUM_PTC_BUFFERS 18

#define NEIGHBOR_NULL -1

#define TAG_RANK_MASK 0xFFFF0000
#define TAG_ID_MASK 0x0000FFFF
#define MAX_TRACKED 1000000

#endif  // _CONSTANTS_H_