
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void vector_add(
    const float* A,
    const float* B,
    float* C,
    int N
) {
//calculating global thread ID(tid)
int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

//vector boundary checking
if (tid<N){
    //each thread adds a single element
    C[tid] = A[tid] + B[tid];
}
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
//1000 elements, then we could have 1000 GPU threads

extern "C" void solve(
  const float* A,
   const float* B,
   float* C,
   int N
   ) {
    if (N<=0) return;

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
int main(){
  const int N = 1000;

  std::vector<float> A(N, 20.0f);
  std::vector<float> B(N, 23.0f);
  std::vector<float> C(N);

  float *dA, *dB, *dC;

  cudaMalloc((void**)&dA, N* sizeof(float) );
  cudaMalloc((void**)&dB, N* sizeof(float) );
  cudaMalloc((void**)&dC, N* sizeof(float));

  cudaMemcpy(dA, A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data(), N*sizeof(float), cudaMemcpyHostToDevice);

  solve(dA,dB,dC,N);

  cudaMemcpy(C.data(),dC, N*sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "C[0] = " <<C[0]<<std::endl;
  std::cout<<"C[999] = "<< C[999]<< std::endl;

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return 0;
}
