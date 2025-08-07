#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H
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

#include "rtweekend.h"
#include "sphere.h"
#include "hittable.h"

#include <memory>
#include <vector>

class hittable_list {
   public:
    __device__ hittable_list() {}
    __device__ hittable_list(sphere* object) {
        add(object);
    }

    __device__ void clear() {
        tail = 0;
    }
    __device__ void add(sphere* object) {
        objects[tail++] = object;
    }

    __device__ bool hit(
        const ray& r, double t_min, double t_max, hit_record* rec) const;

   public:
    sphere** objects;
    int tail = 0;
};

__device__ bool hittable_list::hit(const ray& r, double t_min, double t_max, hit_record* rec) const {
    hit_record temp_rec;
    auto hit_anything = false;
    auto closest_so_far = t_max;

    for (int i = 0; i < tail; i++) {
        const auto object = objects[i];
        if (object->hit(r, t_min, closest_so_far, &temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;

            rec->p = temp_rec.p;
            rec->normal = temp_rec.normal;
            rec->mat_ptr = temp_rec.mat_ptr;
            rec->t = temp_rec.t;
            rec->front_face = temp_rec.front_face;
        }
    }

    return hit_anything;
}

#endif
