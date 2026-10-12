# cuda

Small CUDA C++ kernels and PyTorch scripts I wrote while learning GPU
programming. Each kernel covers one common operation from deep learning
workloads and is written in a simple, readable form rather than tuned for
speed.

## Contents

### CUDA kernels

| File | Kernel | What it does |
| --- | --- | --- |
| `2dmatrix.cu` | `matrixAdd_kernel` | Element-wise addition of two 2D matrices |
| `gemm.cu` | `gemm_kernel` | Naive matrix multiply, `C = A * B` |
| `cuda2d.cu` | `conv2d_kernel` | 2D convolution with no padding and stride 1 |
| `maxpooling.cu` | `maxpool2d_kernel` | 2D max pooling with a square, non-overlapping window |
| `softmax.cu` | `softmax_naive_kernel` | Row-wise softmax, using the row max for numerical stability |

All matrices are stored in row-major order. Each kernel uses a 2D grid where
`x` maps to columns and `y` maps to rows, and each thread computes one
output element. Every kernel checks bounds, so input sizes don't need to be
a multiple of the block size.

Notes on individual kernels:

- **gemm**: `A` is `M_rows x K_shared_dim`, `B` is `K_shared_dim x N_cols`
  and `C` is `M_rows x N_cols`.
- **conv2d**: for a `height x width` input and a `kernel_dim x kernel_dim`
  filter, the output is `(height - kernel_dim + 1) x (width - kernel_dim + 1)`.
- **maxpool2d**: the output is `(height / pool_dim) x (width / pool_dim)`.
  Any leftover rows or columns are dropped.
- **softmax**: each thread recomputes the max and the sum for its whole
  row. This keeps the code simple but repeats work. A faster version would
  do one reduction per row in shared memory.

The `.cu` files contain kernels only, with no `main` or host code. To run
one, you need a small driver program that allocates device memory, copies
the inputs over, launches the kernel and copies the result back.

### PyTorch scripts

These scripts train a small MLP on MNIST as a baseline to compare the CUDA
kernels against.

| File | Purpose |
| --- | --- |
| `pytorchsetup.py` | Hyperparameters, loading and normalising the MNIST data, and moving it to the GPU |
| `mlp.py` | A two-layer MLP (784 -> 256 -> 10) with He-uniform initialisation, cross-entropy loss and SGD |
| `traintimed.py` | A training loop for one epoch that records time spent on data loading, the forward pass, the loss, the backward pass and the weight updates |
| `checkgradients.py` | Prints the norm and max of each gradient and warns when gradients are very large or very small |

## Requirements

- An NVIDIA GPU that supports CUDA
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads), which provides `nvcc`
- On Windows: Visual Studio Build Tools with the "Desktop development with
  C++" workload. `nvcc` uses MSVC (`cl.exe`) as its host compiler.
- For the Python scripts: Python 3, NumPy, and a CUDA build of
  [PyTorch](https://pytorch.org/get-started/locally/)

## Building the kernels

To check that a kernel compiles:

```sh
nvcc -c gemm.cu -o gemm.o
```

On Windows, run this from the "x64 Native Tools Command Prompt for VS", or
run `vcvars64.bat` first so that `cl.exe` and the Windows SDK are on your
`PATH`.

Example launch for the GEMM kernel:

```cpp
dim3 block(16, 16);
dim3 grid((N_cols + block.x - 1) / block.x,
          (M_rows + block.y - 1) / block.y);
gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M_rows, N_cols, K_shared_dim);
```

The other kernels are launched the same way, with the grid sized to the
output dimensions.

## Data

`pytorchsetup.py` expects MNIST as raw binary files in a `data/` folder:

| File | Type | Shape |
| --- | --- | --- |
| `data/X_train.bin` | `float32` | 60000 x 784 |
| `data/y_train.bin` | `int32` | 60000 |
| `data/X_test.bin` | `float32` | 10000 x 784 |
| `data/y_test.bin` | `int32` | 10000 |

Pixel values should be scaled to `[0, 1]`. The script then normalises them
using the standard MNIST mean (0.1307) and standard deviation (0.3081).
The data files are not included in this repository.
