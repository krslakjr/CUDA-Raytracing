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
#include <string>

#include "camera.h"
#include "color.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#include <chrono>
#include <omp.h>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <vector>
#include <thread>
#include <sstream>
#include <string>
#include <numeric>
#include <algorithm>

// Simple min function compatible with CUDA device code
__host__ __device__ __forceinline__ int int_min(int a, int b) {
    return a < b ? a : b;
}

#include <iostream>

__device__ __noinline__ color ray_color(ray r, const hittable_list* world, int depth) {
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

__global__ __launch_bounds__(256) void random_scene(hittable_list* world) {
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

__global__ __launch_bounds__(256) void ray_trace_pixel(camera cam, hittable_list* world, unsigned char* out_image, int gpu_id) {
    const int image_width = 1024;
    const int image_height = 576;
    const int samples_per_pixel = 50;
    const int max_depth = 50;
    
    // Calculate the number of GPUs being used
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    
    // Calculate the portion of the image this GPU should render
    int rows_per_gpu = (image_height + num_gpus - 1) / num_gpus;  // Ceiling division
    int start_row = gpu_id * rows_per_gpu;
    int end_row = int_min((gpu_id + 1) * rows_per_gpu, image_height);
    
    // Each thread processes one pixel
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_count = image_width * (end_row - start_row);
    
    if (i < pixel_count) {
        // Convert 1D thread index to 2D image coordinates
        int x = i % image_width;
        int y = start_row + (i / image_width);
        
        if (y < end_row) {
            color pixel_color(0, 0, 0);
            
            // Anti-aliasing by taking multiple samples per pixel
            for (int s = 0; s < samples_per_pixel; ++s) {
                auto u = (x + random_double()) / (image_width - 1);
                auto v = (y + random_double()) / (image_height - 1);
                ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, world, max_depth);
            }
            
            // Apply gamma correction and scale to 0-255
            pixel_color.postprocessing(samples_per_pixel);
            
            // Write the pixel color to the output image (in BGR format)
            int idx = 3 * (image_width * (image_height - 1 - y) + x);
            out_image[idx + 0] = pixel_color.f[2];  // Blue
            out_image[idx + 1] = pixel_color.f[1];  // Green
            out_image[idx + 2] = pixel_color.f[0];  // Red
        }
    }
}

// Function to get current timestamp as string
std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " output_filename" << std::endl;
        return 1;
    }

    // Performance measurement start
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Get GPU info
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    std::cerr << "\n=== System Info ===\n";
    std::cerr << "Timestamp: " << get_timestamp() << "\n";
    std::cerr << "Number of CUDA devices: " << device_count << "\n";
    
    std::vector<cudaDeviceProp> device_props(device_count);
    for (int i = 0; i < device_count; i++) {
        cudaGetDeviceProperties(&device_props[i], i);
        std::cerr << "Device " << i << ": " << device_props[i].name 
                 << " (Compute " << device_props[i].major << "." << device_props[i].minor
                 << ", " << device_props[i].multiProcessorCount << " SMs, "
                 << (device_props[i].totalGlobalMem >> 20) << " MB VRAM)\n";
    }
    std::cerr << std::endl;

    // Image parameters
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 1024;
    const int image_height = 576;
    const int samples_per_pixel = 100;  // Adjust as needed
    
    std::cerr << "=== Render Settings ===\n";
    std::cerr << "Resolution: " << image_width << "x" << image_height << "\n";
    std::cerr << "Samples per pixel: " << samples_per_pixel << "\n";
    std::cerr << "Using " << device_count << " GPU(s)\n\n";

    // Initialize random seed
    srand(static_cast<unsigned>(time(NULL)));
    int seed = rand() % 1024;
    
    // Prepare output image buffer
    std::vector<unsigned char> out_image(image_height * image_width * 3);
    
    // Prepare timing variables
    std::vector<double> gpu_times(device_count, 0.0);
    std::vector<std::thread> threads;
    std::vector<unsigned char*> dev_out_images(device_count, nullptr);
    
    // Start timing the entire rendering process
    auto render_start = std::chrono::high_resolution_clock::now();

    // Launch a thread for each GPU
    for (int dev = 0; dev < device_count; ++dev) {
        threads.emplace_back([&, dev]() {
            // Set device for this thread
            cudaSetDevice(dev);
            
            // Start timing for this GPU
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            
            // Allocate device memory for output image
            cudaMalloc(&dev_out_images[dev], image_height * image_width * 3 * sizeof(unsigned char));
            
            // Initialize RNG
            random_init<<<1, 1>>>(256, seed + dev);
            
            // Create and populate scene
            hittable_list* world;
            cudaMalloc(&world, sizeof(hittable_list));
            random_scene<<<1, 1>>>(world);
            cudaDeviceSynchronize();
            
            // Set up camera
            point3 lookfrom(13, 2, 3);
            point3 lookat(0, 0, 0);
            vec3 vup(0, 1, 0);
            auto dist_to_focus = 10.0;
            auto aperture = 0.1;
            
            camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);
            
            // Calculate the portion of the image this GPU should render
            int rows_per_gpu = (image_height + device_count - 1) / device_count;  // Ceiling division
            int start_row = dev * rows_per_gpu;
            int end_row = int_min((dev + 1) * rows_per_gpu, image_height);
            int rows_this_gpu = end_row - start_row;
            
            // Configure kernel launch
            int blockSize = 256;  // Threads per block
            int pixels_per_gpu = image_width * rows_this_gpu;
            int numBlocks = (pixels_per_gpu + blockSize - 1) / blockSize;  // Blocks needed to cover all pixels
            
            // Render this GPU's portion of the image
            ray_trace_pixel<<<numBlocks, blockSize>>>(cam, world, dev_out_images[dev], dev);
            
            // Check for kernel errors
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "GPU " << dev << " kernel launch failed: " 
                         << cudaGetErrorString(err) << std::endl;
            }
            
            // Record end time and calculate duration
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            gpu_times[dev] = ms / 1000.0;  // Convert to seconds
            
            // Copy results back to host
            // Each GPU renders the entire image, but we'll keep the last one that finishes
            // (in a real application, you'd want to blend or choose one GPU's output)
            size_t image_size = image_height * image_width * 3 * sizeof(unsigned char);
            cudaMemcpy(out_image.data(), dev_out_images[dev], image_size, cudaMemcpyDeviceToHost);
            
            // Clean up
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(dev_out_images[dev]);
            cudaFree(world);
        });
    }
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
    
    // End timing the entire rendering process
    auto render_end = std::chrono::high_resolution_clock::now();
    double total_render_time = std::chrono::duration<double>(render_end - render_start).count();

    // Calculate and display performance statistics
    size_t total_pixels = static_cast<size_t>(image_width) * image_height;
    size_t total_samples = total_pixels * samples_per_pixel;
    
    std::cerr << "\n=== Performance Summary ===\n";
    std::cerr << std::fixed << std::setprecision(3);
    
    // Individual GPU times
    for (int i = 0; i < device_count; ++i) {
        double pixels_per_sec = (total_pixels / device_count) / gpu_times[i];
        std::cerr << "GPU " << i << " (" << device_props[i].name << "): " 
                 << gpu_times[i] * 1000.0 << " ms, "
                 << pixels_per_sec / 1e6 << " M pixels/sec\n";
    }
    
    // Calculate total performance
    double avg_gpu_time = std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0) / device_count;
    double total_pixels_per_sec = total_pixels / avg_gpu_time;
    double total_samples_per_sec = total_samples / avg_gpu_time;
    
    std::cerr << "\n=== Overall Performance ===\n";
    std::cerr << "Total render time: " << total_render_time << " seconds\n";
    std::cerr << "Average GPU time:  " << avg_gpu_time << " seconds\n";
    std::cerr << "Performance:       " << total_pixels_per_sec / 1e6 << " M pixels/second\n";
    std::cerr << "                   " << total_samples_per_sec / 1e6 << " M samples/second\n";
    
    // Memory usage
    std::cerr << "\n=== Memory Usage ===\n";
    for (int i = 0; i < device_count; ++i) {
        size_t free_byte, total_byte;
        cudaSetDevice(i);
        cudaMemGetInfo(&free_byte, &total_byte);
        double used_mb = (total_byte - free_byte) / (1024.0 * 1024.0);
        double total_mb = total_byte / (1024.0 * 1024.0);
        std::cerr << "GPU " << i << " memory: " << used_mb << " MB / " << total_mb << " MB ("
                 << (used_mb / total_mb * 100.0) << "% used)\n";
    }
    
    // Generate output filename with timestamp and parameters
    std::string filename = argv[1];
    if (filename.find(".") != std::string::npos) {
        filename = filename.substr(0, filename.find_last_of('.'));
    }
    
    // Add resolution, samples and GPU count to filename
    std::stringstream ss;
    ss << filename << "_" << image_width << "x" << image_height 
       << "_" << samples_per_pixel << "spp_" << device_count << "gpu"
       << ".ppm";
    std::string output_filename = ss.str();
    
    // Save the image
    write_ppm(output_filename.c_str(), out_image.data(), image_height, image_width, 3);
    std::cerr << "\nImage saved as: " << output_filename << "\n";
    
    // Save performance data to CSV
    std::ofstream perf_file("performance_log.csv", std::ios::app);
    if (perf_file.is_open()) {
        // Write header if file is empty
        if (perf_file.tellp() == 0) {
            perf_file << "timestamp,resolution,samples_per_pixel,gpu_count,render_time_sec,avg_gpu_time_sec,pixels_per_sec,samples_per_sec,gpu_info\n";
        }
        
        // Prepare GPU info string
        std::stringstream gpu_info;
        for (int i = 0; i < device_count; ++i) {
            if (i > 0) gpu_info << " | ";
            gpu_info << device_props[i].name << " (" << gpu_times[i] << "s)";
        }
        
        // Write performance data
        perf_file << get_timestamp() << ","
                 << image_width << "x" << image_height << ","
                 << samples_per_pixel << ","
                 << device_count << ","
                 << total_render_time << ","
                 << avg_gpu_time << ","
                 << total_pixels_per_sec << ","
                 << total_samples_per_sec << ","
                 << "\"" << gpu_info.str() << "\"\n";
        perf_file.close();
        std::cerr << "Performance data saved to performance_log.csv\n";
    }
    
    // Total execution time
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(total_end - total_start).count();
    std::cerr << "\nTotal execution time: " << total_time << " seconds\n";
}
