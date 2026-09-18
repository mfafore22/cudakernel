# cuda

A collection of small CUDA C++ learning exercises.

## Files

- `gemm.cu` — `gemm_kernel`, a naive general matrix multiply (GEMM) kernel
  computing `C = A * B` for row-major matrices:
  - `A` is `M_rows x K_shared_dim`
  - `B` is `K_shared_dim x N_cols`
  - `C` is `M_rows x N_cols`

  Each thread computes one element of `C` (2D grid/block; `x` maps to
  columns, `y` maps to rows) by accumulating the dot product of a row of
  `A` with a column of `B`. Bounds checks handle sizes that aren't a
  multiple of the block size. This is kernel only: there is no host driver
  code yet.

## Example launch

```cpp
dim3 block(16, 16);
dim3 grid((N_cols + block.x - 1) / block.x,
          (M_rows + block.y - 1) / block.y);
gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M_rows, N_cols, K_shared_dim);
```

## Requirements

- An NVIDIA GPU (CUDA-capable)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (provides `nvcc`)
- On Windows: MSVC build tools + Windows SDK (Visual Studio Build Tools),
  since `nvcc` uses `cl.exe` as its host compiler

## Build

`gemm.cu` has no `main`, so it needs a driver program to run. To check that
it compiles:

```sh
nvcc -c gemm.cu -o gemm.o
```

On Windows, run this from a "x64 Native Tools Command Prompt for VS" (or
after running `vcvars64.bat`) so `cl.exe` and the Windows SDK headers are
on `PATH`.
