# CUDA Vector Addition

## 1. Problem Overview

The task is to add two input vectors, `A` and `B`, element by element and store the result in vector `C`.

For two vectors of length `N`:

$$
C[i] = A[i] + B[i], \qquad 0 \le i < N
$$

Each element is a 32-bit floating-point value (`float`). The input vectors have the same length, and the program must write every result to the corresponding position in `C`.

### Example

```text
A = [1.0, 2.0, 3.0, 4.0]
B = [5.0, 6.0, 7.0, 8.0]

C = [6.0, 8.0, 10.0, 12.0]
```

The operation is independent for every index. Calculating `C[0]` does not depend on calculating `C[1]`, so the elements can be processed in parallel on a GPU.

### Constraints and benchmark

- `1 <= N <= 100,000,000`
- `A` and `B` have identical lengths.
- Elements are 32-bit floating-point numbers.
- Performance is measured with `N = 25,000,000`.
- The required output must be stored in `C`.
- The `solve` function signature must remain unchanged.
- External libraries are not permitted.

---

## 2. Solution Structure

The implementation has three main parts:

1. **CUDA kernel (`vector_add`)** – runs on the GPU. Each active GPU thread calculates one element of `C`.
2. **Host wrapper (`solve`)** – runs on the CPU. It calculates the CUDA launch dimensions, launches the kernel, and waits for the GPU work to finish.
3. **Test program (`main`)** – used for local testing. It creates the input vectors, allocates GPU memory, copies data to the GPU, calls `solve`, copies the results back, prints selected values, and releases GPU memory.

The data flow is:

```text
CPU memory                         GPU memory
-----------                        ----------
A (input)  -- cudaMemcpy --------> dA
B (input)  -- cudaMemcpy --------> dB
                                      |
                                      v
                              vector_add kernel
                              C[i] = A[i] + B[i]
                                      |
                                      v
C (output) <-- cudaMemcpy -------- dC
```

The `d` prefix is a naming convention used here to indicate a pointer to device (GPU) memory. It is not a special CUDA keyword.

---

## 3. CUDA Kernel: `vector_add`

```cpp
__global__ void vector_add(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}
```

### What the components mean

| Component | Meaning |
|---|---|
| `__global__` | CUDA function qualifier. It declares a kernel that is launched from the CPU and executed on the GPU. |
| `const float* A` | Pointer to the first input vector in GPU memory. `const` means this kernel does not modify the values in `A`. |
| `const float* B` | Pointer to the second input vector in GPU memory. The kernel does not modify it. |
| `float* C` | Pointer to the output vector in GPU memory. The kernel writes the sums here. |
| `int N` | Number of elements in each vector. |
| `blockIdx.x` | Index of the current block in the grid. |
| `blockDim.x` | Number of threads in each block. |
| `threadIdx.x` | Index of the current thread within its block. |

### Calculating the global thread ID

```cpp
int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
```

CUDA organizes launched threads into blocks, and blocks into a grid. `threadIdx.x` is only the thread's local index within its block, so it is not enough by itself to identify an element across the entire grid.

The expression combines the block's starting offset with the thread's local index:

$$
\text{tid} = (\text{blockIdx.x} \times \text{blockDim.x}) + \text{threadIdx.x}
$$

For example, with 256 threads per block:

- Block 0 contains global thread IDs `0` through `255`.
- Block 1 contains global thread IDs `256` through `511`.
- Block 2 contains global thread IDs `512` through `767`.
- Block 3 contains global thread IDs `768` through `1023`.

### Boundary check

```cpp
if (tid < N) {
    C[tid] = A[tid] + B[tid];
}
```

The number of launched threads is rounded up to a whole number of blocks. Consequently, the final block can contain threads whose IDs are greater than or equal to `N`.

The condition `tid < N` prevents those threads from accessing memory outside the vectors. Threads with valid IDs perform one addition; threads with invalid IDs do nothing.

The actual element-wise operation is:

```cpp
C[tid] = A[tid] + B[tid];
```

Each active thread reads one value from `A`, reads the value at the same index from `B`, adds them, and writes the result to `C` at that index.

---

## 4. Host Wrapper: `solve`

```cpp
extern "C" void solve(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    if (N <= 0) return;

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```

### `extern "C"`

`extern "C"` requests C language linkage for the function name. This commonly appears in GPU coding environments where the host-side function is called by a harness or another component that expects an unmangled function name. Keep the required `solve` signature exactly as specified by the problem environment.

### Handling the input size

```cpp
if (N <= 0) return;
```

This exits when there are no elements to process. The stated problem constraint has `N >= 1`, but the check makes the wrapper safer for other callers.

### Threads per block

```cpp
int threadsPerBlock = 256;
```

This sets the number of threads in each CUDA block to 256. It is a launch configuration choice, not a statement that the whole GPU has only 256 threads. Many blocks can be launched, and the GPU schedules them on its available hardware.

### Calculating the number of blocks

```cpp
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
```

This is integer arithmetic for the ceiling division:

$$
\text{blocksPerGrid} = \left\lceil \frac{N}{256} \right\rceil
$$

It chooses enough blocks to provide at least `N` thread positions. The subtraction by one makes the expression work for both exact and non-exact multiples of 256.

For example, when `N = 1000`:

$$
\text{blocksPerGrid} = \left\lceil \frac{1000}{256} \right\rceil = 4
$$

Four blocks launch `4 * 256 = 1024` threads. Threads with IDs `0` through `999` process vector elements. Threads `1000` through `1023` fail the boundary check and do no work.

### Launching the kernel

```cpp
vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
```

The CUDA launch syntax is:

```cpp
kernel_name<<<number_of_blocks, threads_per_block>>>(arguments...);
```

Here, the wrapper launches `vector_add` with the calculated grid and block dimensions and passes the GPU pointers and vector length to the kernel.

### Synchronizing

```cpp
cudaDeviceSynchronize();
```

Kernel launches are generally asynchronous with respect to the CPU. This call makes the CPU wait until previously issued work on the device has completed. In this example, it ensures the kernel has finished before `solve` returns.

For production-quality code, CUDA API calls and kernel launches should also be checked for errors. The short example omits that error-handling code to keep the core operation clear.

---

## 5. Local Test Program: `main`

The `main` function is a local test harness. It is useful for compiling and running the program yourself, but it is normally **not included in a submission** when the platform supplies its own harness and requires only the `solve` function.

### Creating the input and output vectors

The explicit four-element test is:

```cpp
const int N = 4;

std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
std::vector<float> B = {5.0f, 6.0f, 7.0f, 8.0f};
std::vector<float> C(N);
```

- `N` is the number of elements.
- `A` and `B` contain the input values.
- `C(N)` creates an output vector with space for `N` floats.

The expected result is:

```text
C = [6, 8, 10, 12]
```

The larger local test uses:

```cpp
const int N = 1000;

std::vector<float> A(N, 20.0f);
std::vector<float> B(N, 23.0f);
std::vector<float> C(N);
```

The two-argument vector constructor fills every element with the supplied value. Therefore, this test creates 1,000 copies of `20.0f` in `A` and 1,000 copies of `23.0f` in `B`. Every output element should be `43.0f`.

This is a useful test of a larger vector and the grid calculation, but it does not test varied input values. The four-element test checks a different input value at each index.

### Allocating device memory

```cpp
float *dA, *dB, *dC;

cudaMalloc((void**)&dA, N * sizeof(float));
cudaMalloc((void**)&dB, N * sizeof(float));
cudaMalloc((void**)&dC, N * sizeof(float));
```

`std::vector` stores the test data in CPU (host) memory. The kernel needs pointers to GPU (device) memory, so the program allocates three device buffers:

- `dA` for a copy of input vector `A`.
- `dB` for a copy of input vector `B`.
- `dC` for the output values.

`sizeof(float)` gives the number of bytes occupied by one `float`, which is normally 4. Thus, each allocation requests `N * 4` bytes.

`cudaMalloc` allocates memory on the GPU. The cast to `(void**)` matches the API's pointer-to-pointer parameter.

### Copying input data to the GPU

```cpp
cudaMemcpy(dA, A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(dB, B.data(), N * sizeof(float), cudaMemcpyHostToDevice);
```

`A.data()` and `B.data()` return pointers to the vectors' contiguous CPU-side element storage. `cudaMemcpy` transfers the input bytes from host memory to device memory.

The direction `cudaMemcpyHostToDevice` means CPU memory to GPU memory.

### Calling `solve`

```cpp
solve(dA, dB, dC, N);
```

The test harness passes the device pointers, not the original host pointers. The wrapper launches the GPU kernel, which reads `dA` and `dB` and writes the output into `dC`.

### Copying the result back to the CPU

```cpp
cudaMemcpy(C.data(), dC, N * sizeof(float), cudaMemcpyDeviceToHost);
```

This copies the output from GPU memory into the CPU-side vector `C`. The direction `cudaMemcpyDeviceToHost` means GPU memory to CPU memory.

### Printing and releasing memory

```cpp
std::cout << "C[0] = " << C[0] << std::endl;
std::cout << "C[999] = " << C[999] << std::endl;

cudaFree(dA);
cudaFree(dB);
cudaFree(dC);
```

The `N = 1000` test prints the first and last output elements. Both should be `43`, because every input pair is `20 + 23`.

`cudaFree` releases each GPU allocation after it is no longer needed. The local test should check that allocation and copy operations succeed before relying on their results; those checks are omitted from this instructional example.

---

## 6. Components and APIs Used

| Component / API | Role in this program |
|---|---|
| `cuda_runtime.h` | Provides CUDA Runtime API declarations used by the program. |
| `iostream` | Provides `std::cout` and `std::endl` for printing test output. |
| `vector` | Provides `std::vector<float>` for storing host-side input and output arrays in the test harness. |
| `float` | 32-bit floating-point type used for the vector elements. |
| `__global__` | Marks `vector_add` as a GPU kernel callable from host code. |
| `blockIdx.x` | Identifies the block containing the current thread. |
| `blockDim.x` | Gives the number of threads in a block. |
| `threadIdx.x` | Identifies the current thread inside its block. |
| `cudaMalloc` | Allocates buffers in GPU memory. |
| `cudaMemcpy` | Copies data between CPU and GPU memory. |
| `cudaMemcpyHostToDevice` | Selects CPU-to-GPU transfer direction. |
| `cudaMemcpyDeviceToHost` | Selects GPU-to-CPU transfer direction. |
| `<<<...>>>` | CUDA kernel launch configuration syntax. |
| `cudaDeviceSynchronize` | Waits for the GPU work to complete. |
| `cudaFree` | Releases GPU memory allocations. |
| `extern "C"` | Gives the `solve` function C language linkage. |

No external numerical or parallel-processing library is used. The GPU computation is written directly using CUDA C++ and the CUDA Runtime API.

---

## 7. How the Launch Scales Across Different Values of `N`

With 256 threads per block, the grid size changes with `N`:

| Vector length (`N`) | Blocks launched | Threads launched | Threads that perform additions | Threads that skip the boundary check |
|---:|---:|---:|---:|---:|
| 4 | 1 | 256 | 4 | 252 |
| 1,000 | 4 | 1,024 | 1,000 | 24 |
| 25,000,000 | 97,657 | 25,000,192 | 25,000,000 | 192 |
| 100,000,000 | 390,625 | 100,000,000 | 100,000,000 | 0 |

The table shows why the boundary check is necessary: the total number of launched threads may be slightly larger than `N`.

For the benchmark size `N = 25,000,000`, each vector occupies:

\[
25,000,000 \times 4 = 100,000,000\text{ bytes}
\]

The three GPU buffers (`dA`, `dB`, and `dC`) therefore require approximately 300 MB in total, using decimal megabytes. This is the device memory required by the arrays alone, excluding runtime and other allocations.

---

## 8. Important Notes and Corrections

1. **`N = 4` and `N = 1000` are local test sizes.** They help validate correctness and indexing. They are not the performance benchmark size.
2. **`N = 1000` does not launch exactly 1,000 threads.** With 256 threads per block, it launches 4 blocks, or 1,024 threads. The final 24 threads are safely ignored by `if (tid < N)`.
3. **The `N = 1000` test uses repeated values.** Since `A` is filled with `20` and `B` with `23`, every element of `C` should be `43`. To test varied inputs, use explicit arrays or generate different values for each index.
4. **The benchmark is much larger.** `N = 25,000,000` requires 97,657 blocks at 256 threads per block. The same kernel and launch formula handle this size; no fixed array length is hard-coded in the kernel.
5. **Keep the submission interface unchanged.** If the judge provides its own `main` or test harness, submit the required `solve` implementation rather than a second `main` function.
6. **`%%writefile vectorAddition.cu` is notebook syntax.** It is used in environments such as Colab or Kaggle to write the following cell contents into a file. It is not a C++ or CUDA statement and should not appear inside the `.cu` source file itself.
7. **Check CUDA errors in a robust implementation.** The teaching code does not inspect the return values of `cudaMalloc`, `cudaMemcpy`, `cudaDeviceSynchronize`, or the kernel launch. Error checking is important when diagnosing allocation, launch, or execution failures.

---

## 9. Core Idea

The solution assigns one GPU thread to each vector index. A thread computes its global index, verifies that the index is within the vector length, and performs one floating-point addition. The CPU wrapper calculates enough blocks to cover all elements and launches the kernel. The local test harness manages host and device memory and verifies the returned values.

The essential computation is only:

```cpp
C[tid] = A[tid] + B[tid];
```

The rest of the program establishes the GPU execution layout and moves data between CPU and GPU memory.
