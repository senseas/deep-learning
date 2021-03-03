#pragma OPENCL EXTENSION cl_intel_printf : enable
// let C = A * B
// M : number of rows in A
// N : number of rows in B (also number of columns in A)
// P : number of columns in B
kernel void matmul(global const float *A, global const float *B, global float *C, int M, int N, int P) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  float sum = 0;

  int s = x * N;
  for (int i = 0; i < N; i++) {
    sum += A[s + i] * B[y + i * P];
  }
  //printf("%f\n",  sum);
  C[x * P + y] = sum;
}

// let C = A * B
// M : number of rows in A (must can be divisible by 8)
// N : number of rows in B (also number of columns in A) (must can be divisible by 8)
// P : number of columns in B
// this function is faster than the one above
#define WORK_ITEM_M 8
#define WORK_ITEM_N 8
kernel void matmulx(global const float *A, global const float *B, global float *C, int M, int N, int P, int maxMId, int maxNId) {
  if (get_global_id(0) >= maxMId) return;
  if (get_global_id(1) >= maxNId) return;
  int x = get_global_id(0) * WORK_ITEM_M;
  int y = get_global_id(1) * WORK_ITEM_N;

  local float sum[WORK_ITEM_M][WORK_ITEM_N];
  float data1[WORK_ITEM_M];
  float data2[WORK_ITEM_N];

  for (int m = 0; m < WORK_ITEM_M; m++) {
    for (int n = 0; n < WORK_ITEM_N; n++) {
      sum[m][n] = 0;
    }
  }

  for (int n = 0; n < N; n++) {
    for (int i = 0; i < WORK_ITEM_M; i++) {
      data1[i] = A[(x + i) * N + n];
    }

    for (int i = 0; i < WORK_ITEM_N; i++) {
      data2[i] = B[y + n * P + i];
    }

    for (int m = 0; m < WORK_ITEM_M; m++) {
      for (int n = 0; n < WORK_ITEM_N; n++) {
        sum[m][n] += data1[m] * data2[n];
      }
    }
  }

  for (int m = 0; m < WORK_ITEM_M; m++) {
    for (int n = 0; n < WORK_ITEM_N; n++) {
      C[(x + m) * P + y + n] = sum[m][n];
    }
  }
}