Parallel Ray Tracing with CUDA
===

Implement the parallel ray tracing algorithm with CUDA.

Build
---
```bash
$ make
```

Run
---
- Single GPU version
  ```bash
  $ ./main output.png
  ```
- Multi-GPU version
  ```bash
  $ ./main_multi output.png
  ```

Demonstration
---

### samples per pixel = 100
![high quality](demo2.png)

### samples per pixel = 20
![low quality](demo.png)


Acknowledgement
---
The source code of sequential raytracing algorithm is based on ![raytracing.github.io](https://github.com/RayTracing/raytracing.github.io/).

Many thanks to my teammate ![Matt Yao](https://github.com/JNNNNYao).
