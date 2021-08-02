#pragma OPENCL EXTENSION cl_intel_printf : enable
// let C = A * B
// M : number of rows in A
// N : number of rows in B (also number of columns in A)
// P : number of columns in B
kernel void Matmul(
  global const float *A, global const float *B,
  global float *C, int M, int N, int H) {

  int x = get_global_id(0), y = get_global_id(1), s = x * H;

  float sum = 0;

  for (int i = 0; i < H; i++) {

    sum += A[s + i] * B[y + i * N];

  }

  C[x * N + y] = sum;

}

// let C = A * B
// M : number of rows in A
// N : number of rows in B (also number of columns in A)
// P : number of columns in B
kernel void MatmulGradient(
  global const float *A, global const float *B,
  global float *DA, global float *DB,
  global float *DC, int M, int N, int H) {

  int x = get_global_id(0), y = get_global_id(1), s = x * H;

  float grad = DC[x * N + y];

  for (int i = 0; i < H; i++) {

    DA[s + i] = grad * B[y + i * N];

    DB[y + i * N] = grad * A[s + i];

  }

}