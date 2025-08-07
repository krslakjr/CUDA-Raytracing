//==============================================================================================
// Parallel computing version by Sky Liu and Matt Yao in 2021-2022.
//
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

#include "camera.h"
#include "color.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#include <chrono>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>

__device__ color ray_color(ray r, const hittable_list* world, int depth) {
    hit_record rec;
    color accu(1, 1, 1);  // accumulation of attenuation
    for (int i = depth; i > 0; i--) {
        if (world->hit(r, 0.001, RT_INFINITY, &rec)) {
            ray scattered;
            color attenuation;
            if (rec.mat_ptr->scatter(r, rec, &attenuation, &scattered)) {
                accu = accu * attenuation;
                r = scattered;
                continue;
            }
            return color(0, 0, 0);
        } else {
            vec3 unit_direction = unit_vector(r.direction());
            auto t = 0.5 * (unit_direction.y() + 1.0);
            return accu * ((1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0));
        }
    }

    return color(0, 0, 0);
}

__global__ void random_scene(hittable_list* world) {
    world->objects = new sphere*[500];
    world->tail = 0;

    auto ground_material = new material(1);
    ground_material->setup1(color(0.5, 0.5, 0.5));
    world->add(new sphere(point3(0, -1000, 0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                material* sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = new material(1);
                    sphere_material->setup1(albedo);
                    world->add(new sphere(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = new material(2);
                    sphere_material->setup2(albedo, fuzz);
                    world->add(new sphere(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = new material(3);
                    sphere_material->setup3(1.5);
                    world->add(new sphere(center, 0.2, sphere_material));
                }
            }
        }
    }
    auto material1 = new material(3);
    material1->setup3(1.5);
    world->add(new sphere(point3(0, 1, 0), 1.0, material1));

    auto material2 = new material(1);
    material2->setup1(color(0.4, 0.2, 0.1));
    world->add(new sphere(point3(-4, 1, 0), 1.0, material2));

    auto material3 = new material(2);
    material3->setup2(color(0.7, 0.6, 0.5), 0.0);
    world->add(new sphere(point3(4, 1, 0), 1.0, material3));
}

__global__ void ray_trace_pixel(camera cam, hittable_list* world, unsigned char* out_image) {

    const int image_width = 1024;
    const int image_height = 576;
    const int samples_per_pixel = 20;
    const int max_depth = 50;

#pragma unroll
    for (int k = 0; k < 4; k++) {
        int i = threadIdx.x * 4 + k, j = blockIdx.x;
        color pixel_color(0, 0, 0);
        for (int s = 0; s < samples_per_pixel; ++s) {
            auto u = (i + random_double()) / (image_width - 1);
            auto v = (j + random_double()) / (image_height - 1);
            ray r = cam.get_ray(u, v);
            pixel_color += ray_color(r, world, max_depth);
        }

        pixel_color.postprocessing(samples_per_pixel);
        out_image[3 * (image_width * (image_height - 1 - j) + i) + 0] = pixel_color.f[2];
        out_image[3 * (image_width * (image_height - 1 - j) + i) + 1] = pixel_color.f[1];
        out_image[3 * (image_width * (image_height - 1 - j) + i) + 2] = pixel_color.f[0];
    }
}

int main(int argc, char** argv) {
    // Performance measurement start
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Get GPU info
    int device_count;
    cudaGetDeviceCount(&device_count);
    std::cerr << "\n=== System Info ===\n";
    std::cerr << "Number of CUDA devices: " << device_count << "\n";
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cerr << "Device " << i << ": " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")\n";
    }
    std::cerr << std::endl;

    // Image
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 1024;
    const int image_height = 576;

    unsigned char* out_image = (unsigned char*)malloc(image_height * image_width * 3 * sizeof(unsigned char));
    unsigned char* dev_out_image;
    cudaMalloc(&dev_out_image, image_height * image_width * 3 * sizeof(unsigned char));

    srand(time(NULL));
    random_init<<<1, 1>>>(256, rand() % 1024);

    hittable_list* world;
    cudaMalloc(&world, sizeof(hittable_list));
    random_scene<<<1, 1>>>(world);
    cudaDeviceSynchronize();

    // Camera

    point3 lookfrom(13, 2, 3);
    point3 lookat(0, 0, 0);
    vec3 vup(0, 1, 0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;

    camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

    // Render
    std::cerr << "=== Starting Render ===\n";
    std::cerr << "Resolution: " << image_width << "x" << image_height << " pixels\n";
    std::cerr << "Samples per pixel: " << samples_per_pixel << "\n";
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Start GPU timing
    cudaEventRecord(start);
    auto cpu_start = std::chrono::high_resolution_clock::now();

    // Launch kernel
    std::cerr << "Launching kernel...\n";
    int blockSize = 256;
    int numBlocks = (image_height + blockSize - 1) / blockSize;
    ray_trace_pixel<<<numBlocks, blockSize>>>(cam, world, dev_out_image);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    cudaDeviceSynchronize();
    
    // End GPU timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    
    // Calculate performance metrics
    size_t total_pixels = image_width * image_height;
    size_t total_samples = total_pixels * samples_per_pixel;
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    std::cerr << "\n=== Performance Metrics ===\n";
    std::cerr << std::fixed << std::setprecision(2);
    std::cerr << "GPU Time: " << gpu_ms << " ms\n";
    std::cerr << "CPU Time: " << cpu_time << " ms\n";
    std::cerr << "Total samples: " << total_samples << "\n";
    std::cerr << "Performance: " << (total_pixels / (gpu_ms / 1000.0)) / 1e6 << " M pixels/second\n";
    std::cerr << "            " << (total_samples / (gpu_ms / 1000.0)) / 1e6 << " M samples/second\n";
    
    // Memory usage
    size_t free_byte, total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    double free_mb = free_byte / (1024.0 * 1024.0);
    double total_mb = total_byte / (1024.0 * 1024.0);
    std::cerr << "\n=== Memory Usage ===\n";
    std::cerr << "GPU Memory: " << (total_mb - free_mb) << " MB / " << total_mb << " MB\n";
    
    // Total time
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    std::cerr << "\nTotal execution time: " << total_time << " ms\n";
    
    // Save performance data to CSV
    std::ofstream perf_file("performance.csv", std::ios::app);
    if (perf_file.is_open()) {
        std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        char time_str[100];
        std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        
        perf_file << time_str << ","
                 << image_width << "x" << image_height << ","
                 << samples_per_pixel << ","
                 << gpu_ms << ","
                 << (total_pixels / (gpu_ms / 1000.0)) / 1e6 << ","
                 << (total_samples / (gpu_ms / 1000.0)) / 1e6 << "\n";
        perf_file.close();
    }

    cudaMemcpy(out_image, dev_out_image, image_height * image_width * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Promenite ekstenziju u .ppm
    std::string filename = argv[1];
    if (filename.find(".") != std::string::npos) {
        filename = filename.substr(0, filename.find_last_of('.'));
    }
    filename += ".ppm";
    
    write_ppm(filename.c_str(), out_image, image_height, image_width, 3);
    std::cerr << "\nDone. Slika je sačuvana kao " << filename << "\n";
}
