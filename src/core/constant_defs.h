#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

#include <limits>
#include <math.h>

/// Defines the vector dimension
#define VECTOR_DIM 3

/// Constant pi
#define CONST_PI 3.14159265358979323846

/// MAX_CELL is used to denote empty particles
#define MAX_CELL std::numeric_limits<uint32_t>::max()
#define MAX_TILE std::numeric_limits<uint32_t>::max()
/* #define MAX_CELL 4294967295 */
/* #define MAX_TILE 4294967295 */
// Note: you can only have up to 4 billion cells on one rank!

/// Anything below this is treated as zero
#define EPS 1.0e-10

#define CENTRAL_ZONE 13
#define ZONE_NUMBER 27

#define NUM_BOUNDARIES 6
// Number of particle communication buffers
/* #define NUM_PTC_BUFFERS 18 */

#define NEIGHBOR_NULL -1

#define TAG_RANK_MASK 0xFFFF0000
#define TAG_ID_MASK 0x0000FFFF
#define MAX_TRACKED 1000000

#define RE_SQUARE 7.91402e-26

#define DEPOSIT_EPS 1.0e-10f

#endif  // _CONSTANTS_H_
