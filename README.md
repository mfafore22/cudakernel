# cuda

A collection of small CUDA C++ learning exercises: vector/tensor addition
and image-batch normalization kernels, plus their CPU counterparts.

## Files

- `vecadd.cu` — complete, runnable example. Allocates two arrays of 8
  floats, copies them to device memory, runs a kernel (`vectorAdd`) that
  adds them element-wise across 8 threads in a single block, copies the
  result back, and verifies each element against the CPU-computed
  expected value.
- `2dvector.cu` — `vectorAdd_kernel`, a 1D element-wise vector-add kernel
  (kernel only, no host driver code).
- `3dkernal.cu` — `tensorAdd3D_kernel`, a 3D element-wise tensor-add
  kernel using a 3D grid/block of threads (kernel only).
- `3dtensor.cu` — `tensorAdd3D_cpu`, the CPU reference implementation of
  3D tensor addition (function only).
- `cpu2d.cu` — `matrixAdd_cpu`, the CPU reference implementation of 2D
  matrix addition (function only).
- `batchimage.cu` — `normalizeImageBatch`, a kernel that normalizes a
  batch of images `(N, C, H, W)` by subtracting a mean and dividing by a
  standard deviation (kernel only).

Only `vecadd.cu` currently has the host-side code (memory allocation,
kernel launch, verification) needed to build and run standalone; the
other files are kernel/function snippets meant to be read or dropped
into a driver program.

## Requirements

- An NVIDIA GPU (CUDA-capable)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (provides `nvcc`)
- On Windows: MSVC build tools + Windows SDK (Visual Studio Build Tools),
  since `nvcc` uses `cl.exe` as its host compiler

## Build

```sh
nvcc vecadd.cu -o vecadd
```

On Windows, run this from a "x64 Native Tools Command Prompt for VS" (or
after running `vcvars64.bat`) so `cl.exe` and the Windows SDK headers are
on `PATH`.

## Run

```sh
./vecadd
```

Expected output:

```
All elements are correct.
```
