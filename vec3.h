#ifndef VEC3_H
#define VEC3_H
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include <cmath>
#include <iostream>

using std::fabs;
using std::sqrt;

class vec3 {
   public:
    __host__ __device__ vec3()
        : e{0, 0, 0} {}
    __host__ __device__ vec3(double e0, double e1, double e2)
        : e{e0, e1, e2} {}

    __host__ __device__ double x() const {
        return e[0];
    }
    __host__ __device__ double y() const {
        return e[1];
    }
    __host__ __device__ double z() const {
        return e[2];
    }

    __host__ __device__ vec3 operator-() const {
        return vec3(-e[0], -e[1], -e[2]);
    }
    __host__ __device__ double operator[](int i) const {
        return e[i];
    }
    __host__ __device__ double &operator[](int i) {
        return e[i];
    }

    __host__ __device__ vec3 &operator+=(const vec3 &v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ vec3 &operator*=(const double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ vec3 &operator/=(const double t) {
        return *this *= 1 / t;
    }

    __host__ __device__ double length() const {
        return sqrt(length_squared());
    }

    __host__ __device__ double length_squared() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    __host__ __device__ bool near_zero() const {
        // Return true if the vector is close to zero in all dimensions.
        const auto s = 1e-8;
        return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
    }

    __device__ inline static vec3 random() {
        return vec3(random_double(), random_double(), random_double());
    }

    __device__ inline static vec3 random(double min, double max) {
        return vec3(random_double(min, max), random_double(min, max), random_double(min, max));
    }

    __host__ __device__ inline void postprocessing(int samples_per_pixel) {
        // Replace NaN components with zero. See explanation in Ray Tracing: The Rest of Your Life.
        if (e[0] != e[0]) e[0] = 0.0;
        if (e[1] != e[1]) e[1] = 0.0;
        if (e[2] != e[2]) e[2] = 0.0;

        // Divide the color by the number of samples and gamma-correct for gamma=2.0.
        auto scale = 1.0 / samples_per_pixel;
        e[0] = sqrt(scale * e[0]);
        e[1] = sqrt(scale * e[1]);
        e[2] = sqrt(scale * e[2]);

        f[0] = static_cast<int>(256 * clamp(e[0], 0.0, 0.999));
        f[1] = static_cast<int>(256 * clamp(e[1], 0.0, 0.999));
        f[2] = static_cast<int>(256 * clamp(e[2], 0.0, 0.999));
    }

   public:
    double e[3];
    int f[3];
};

// Type aliases for vec3
using point3 = vec3;  // 3D point
using color = vec3;   // RGB color

// vec3 Utility Functions

inline std::ostream &operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(double t, const vec3 &v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, double t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(vec3 v, double t) {
    return (1 / t) * v;
}

__host__ __device__ inline double dot(const vec3 &u, const vec3 &v) {
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

__device__ inline vec3 random_in_unit_disk() {
    double len = random_double(0, 1);
    double theta = random_double(0, 2);
    vec3 p(len * cospi(theta), len * sinpi(theta), 0);
    return p;
    // while (true) {
    //     auto p = vec3(random_double(-1, 1), random_double(-1, 1), 0);
    //     if (p.length_squared() >= 1) continue;
    //     return p;
    // }
}

__device__ inline vec3 random_in_unit_sphere() {
    // return vec3(0.5, 0.5, 0.5);

    double len = random_double(0, 1);
    double theta = random_double(0, 2);
    double phi = random_double(0, 1);
    vec3 p(len * cospi(theta) * sinpi(phi),  // x
           len * sinpi(theta) * sinpi(phi),  // y
           len * cospi(phi));                // z
    return p;

    // while (true) {
    //     auto p = vec3::random(-1, 1);
    //     if (p.length_squared() >= 1) continue;
    //     return p;
    // }
}

__device__ inline vec3 random_unit_vector() {
    return unit_vector(random_in_unit_sphere());
}

__device__ inline vec3 random_in_hemisphere(const vec3 &normal) {
    vec3 in_unit_sphere = random_in_unit_sphere();
    if (dot(in_unit_sphere, normal) > 0.0)  // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

__host__ __device__ inline vec3 reflect(const vec3 &v, const vec3 &n) {
    return v - 2 * dot(v, n) * n;
}

__host__ __device__ inline vec3 refract(const vec3 &uv, const vec3 &n, double etai_over_etat) {
    auto cos_theta = fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

#endif
