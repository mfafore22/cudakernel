# vecadd

A minimal CUDA C++ example that adds two vectors on the GPU.

## What it does

`vecadd.cu` allocates two arrays of 8 floats, copies them to device memory,
runs a kernel (`vectorAdd`) that adds them element-wise across 8 threads in
a single block, copies the result back, and verifies each element against
the CPU-computed expected value.

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
