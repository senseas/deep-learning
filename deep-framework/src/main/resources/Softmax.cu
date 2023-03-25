extern "C" __global__ void Expx(double* in, double* out) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int M = idx, N = idx;
  out[N + 1] = exp(in[M + 1]);
}
extern "C" __global__ void Sum(double* in, double* out) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int M = idx, N = idx;
  atomicAdd(&out[4], out[N + 1]);
}
extern "C" __global__ void Softmax(double* in, double* out) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int M = idx * 4, N = idx * 6;
  out[N + 0] = exp(in[M + 0]);
  Expx<<<1, 3>>>(in + M, out + N);
  Sum<<<1, 3>>>(in + M, out + N);
  out[N + 5] = out[N + 0] / out[N + 4];
}