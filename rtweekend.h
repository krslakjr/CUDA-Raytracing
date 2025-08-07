#ifndef RTWEEKEND_H
#define RTWEEKEND_H
//==============================================================================================
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <curand_kernel.h>
// Usings

using std::make_shared;
using std::shared_ptr;
using std::sqrt;

// Constants and Macros

#ifdef __CUDACC__
    #define HOST_DEVICE __host__ __device__
#else
    #define HOST_DEVICE
#endif

// Constants as macros for device compatibility
#define RT_INFINITY 1.0e99
#define RT_PI 3.1415926535897932385

__device__ curandState_t *state;

// Utility Functions

HOST_DEVICE inline double degrees_to_radians(double degrees) {
    return degrees * (RT_PI / 180.0);
}

HOST_DEVICE inline double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

__global__ void random_init(int num, int offset) {
    state = new curandState_t[num];
    for (int i = 0; i < num; i++) {
        curand_init(1, i, offset, &state[i]);
    }
}

#ifdef __CUDA_ARCH__
    #define RANDOM_FUNC __device__
    #define RANDOM_SOURCE curand_uniform(&state[threadIdx.x])
#else
    #define RANDOM_FUNC 
    #define RANDOM_SOURCE ((double)rand() / (RAND_MAX + 1.0))
#endif

RANDOM_FUNC inline double random_double() {
    // Returns a random real in [0,1).
    return RANDOM_SOURCE;
}

RANDOM_FUNC inline double random_double(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_double();
}

RANDOM_FUNC inline int random_int(int min, int max) {
    // Returns a random integer in [min,max].
    return static_cast<int>(random_double(min, max+1));
}

// Common Headers

#include "ray.h"
#include "vec3.h"

#endif
