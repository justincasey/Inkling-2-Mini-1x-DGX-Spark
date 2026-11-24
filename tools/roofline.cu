// GB10 roofline: sustained memory bandwidth and dense tensor-core throughput (bf16, fp16, fp8 e4m3).
// Build: nvcc -O3 -arch=sm_121 tools/roofline.cu -o roofline -lcublasLt -lcublas
// Prints METRIC lines. Not part of the serving path; it exists so the targets in docs/ROOFLINE.md rest on measurements.
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("cuda error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t cb_st_ = (x); if (cb_st_ != CUBLAS_STATUS_SUCCESS) { printf("cublas error %d at %d\n", (int)cb_st_, __LINE__); exit(1); } } while (0)

__global__ void read_sum(const uint4 *p, size_t n, unsigned *out) {
  unsigned acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = __ldcs(p + i);
    acc += v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x12345678u) *out = acc;
}

static double bandwidth(size_t bytes) {
  uint4 *p; unsigned *o;
  CK(cudaMalloc(&p, bytes)); CK(cudaMalloc(&o, 4)); CK(cudaMemset(p, 1, bytes));
  size_t n = bytes / 16;
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int w = 0; w < 3; w++) read_sum<<<4096, 256>>>(p, n, o);
  cudaEventRecord(a);
  const int reps = 10;
  for (int r = 0; r < reps; r++) read_sum<<<4096, 256>>>(p, n, o);
  cudaEventRecord(b); CK(cudaEventSynchronize(b));
  float ms; cudaEventElapsedTime(&ms, a, b);
  cudaFree(p); cudaFree(o);
  return (double)bytes * reps / (ms * 1e-3) / 1e9;
}

static double gemm(cublasLtHandle_t h, cudaDataType_t ab, cudaDataType_t cd, int m, int n, int k, const char *name) {
  cublasLtMatmulDesc_t d; CB(cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;  // fp8 needs the TN layout
  CB(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
  CB(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
  cublasLtMatrixLayout_t A, B, C;
  CB(cublasLtMatrixLayoutCreate(&A, ab, k, m, k));
  CB(cublasLtMatrixLayoutCreate(&B, ab, k, n, k));
  CB(cublasLtMatrixLayoutCreate(&C, cd, m, n, m));
  size_t esz = (ab == CUDA_R_8F_E4M3) ? 1 : 2;
  void *a, *b, *c, *ws; size_t wss = 64 << 20;
  CK(cudaMalloc(&a, (size_t)m * k * esz)); CK(cudaMalloc(&b, (size_t)n * k * esz));
  CK(cudaMalloc(&c, (size_t)m * n * 2)); CK(cudaMalloc(&ws, wss));
  CK(cudaMemset(a, 0x3c, (size_t)m * k * esz)); CK(cudaMemset(b, 0x3c, (size_t)n * k * esz));
  float alpha = 1, beta = 0;
  cublasLtMatmulPreference_t pref; CB(cublasLtMatmulPreferenceCreate(&pref));
  CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wss, sizeof wss));
  cublasLtMatmulHeuristicResult_t r; int got = 0;
  cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(h, d, A, B, C, C, pref, 1, &r, &got);
  if (st != CUBLAS_STATUS_SUCCESS || got == 0) { printf("METRIC %s_tflops=unsupported\n", name); return 0; }
  cudaStream_t s; cudaStreamCreate(&s);
  for (int w = 0; w < 5; w++) CB(cublasLtMatmul(h, d, &alpha, a, A, b, B, &beta, c, C, c, C, &r.algo, ws, wss, s));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0, s);
  const int reps = 20;
  for (int i = 0; i < reps; i++) CB(cublasLtMatmul(h, d, &alpha, a, A, b, B, &beta, c, C, c, C, &r.algo, ws, wss, s));
  cudaEventRecord(e1, s); CK(cudaEventSynchronize(e1));
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  double tf = 2.0 * m * n * k * reps / (ms * 1e-3) / 1e12;
  cudaFree(a); cudaFree(b); cudaFree(c); cudaFree(ws);
  return tf;
}

int main() {
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  int smem_optin = 0; cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  printf("device %s sm_%d%d, %d SMs, smem opt-in %d B, L2 %d MiB\n", p.name, p.major, p.minor, p.multiProcessorCount, smem_optin, p.l2CacheSize >> 20);
  printf("METRIC read_gbs=%.1f\n", bandwidth((size_t)8 << 30));
  cublasLtHandle_t h; CB(cublasLtCreate(&h));
  for (int sz : {4096, 8192}) {
    printf("METRIC bf16_tflops_%d=%.1f\n", sz, gemm(h, CUDA_R_16BF, CUDA_R_16BF, sz, sz, sz, "bf16"));
    printf("METRIC fp16_tflops_%d=%.1f\n", sz, gemm(h, CUDA_R_16F, CUDA_R_16F, sz, sz, sz, "fp16"));
    printf("METRIC fp8_tflops_%d=%.1f\n", sz, gemm(h, CUDA_R_8F_E4M3, CUDA_R_16BF, sz, sz, sz, "fp8"));
  }
  // prefill-shaped: 4096 rows against a 6144x2048 and a 16384x6144 weight (the q_a and o_proj shapes at TP1)
  printf("METRIC bf16_tflops_4096x6144x2048=%.1f\n", gemm(h, CUDA_R_16BF, CUDA_R_16BF, 4096, 2048, 6144, "bf16s"));
  printf("METRIC bf16_tflops_4096x6144x4096=%.1f\n", gemm(h, CUDA_R_16BF, CUDA_R_16BF, 4096, 6144, 4096, "bf16s2"));
  // Inkling-2-Mini shapes, labelled rows x K x N like the two above: the dense 4096-row projections (K 2304 -> N 3072 and
  // K 3072 -> N 2304) and the 256-row per-expert GEMMs (K 2304 -> N 1152 and K 1152 -> N 2304)
  printf("METRIC bf16_tflops_4096x2304x3072=%.1f\n", gemm(h, CUDA_R_16BF, CUDA_R_16BF, 4096, 3072, 2304, "bf16m1"));
  printf("METRIC bf16_tflops_4096x3072x2304=%.1f\n", gemm(h, CUDA_R_16BF, CUDA_R_16BF, 4096, 2304, 3072, "bf16m2"));
  printf("METRIC bf16_tflops_256x2304x1152=%.1f\n", gemm(h, CUDA_R_16BF, CUDA_R_16BF, 256, 1152, 2304, "bf16e1"));
  printf("METRIC bf16_tflops_256x1152x2304=%.1f\n", gemm(h, CUDA_R_16BF, CUDA_R_16BF, 256, 2304, 1152, "bf16e2"));
  return 0;
}
