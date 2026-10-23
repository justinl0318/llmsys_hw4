#include <cuda_runtime.h>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <fstream>

#define MAX_DIMS 10
#define TILE 32
#define BLOCK_DIM 1024
#define BASE_THREAD_NUM 32

#define ADD_FUNC       1
#define MUL_FUNC       2
#define ID_FUNC        3
#define NEG_FUNC       4
#define LT_FUNC        5
#define EQ_FUNC        6
#define SIGMOID_FUNC   7
#define RELU_FUNC      8
#define RELU_BACK_FUNC 9
#define LOG_FUNC       10
#define LOG_BACK_FUNC  11
#define EXP_FUNC       12
#define INV_FUNC       13
#define INV_BACK_FUNC  14
#define IS_CLOSE_FUNC  15
#define MAX_FUNC       16
#define POW            17
#define TANH           18

__device__ float fn(int fn_id, float x, float y=0) {
    switch(fn_id) {
      case ADD_FUNC: {
        return x + y;
      }
      case MUL_FUNC: {
        return x * y;
      }
      case ID_FUNC: {
      	return x;
      }
      case NEG_FUNC: {
        return -x;
      }
      case LT_FUNC: {
        if (x < y) {
          return 1.0;
        }
        else {
          return 0.0;
        }
      }
      case EQ_FUNC: {
        if (x == y) {
          return 1.0;
        }
        else {
          return 0.0;
        }
      }
      case SIGMOID_FUNC: {
        if (x >= 0) {
          return 1.0 / (1.0 + exp(-x));
        }
        else {
          return exp(x) / (1.0 + exp(x));
        }
      }
      case RELU_FUNC: {
        return max(x, 0.0);
      }
      case RELU_BACK_FUNC: {
        if (x > 0) {
          return y;
        }
        else {
          return 0.0;
        }
      }
      case LOG_FUNC: {
        return log(x + 1e-6);
      }
      case LOG_BACK_FUNC: {
        return y / (x + 1e-6);
      }
      case EXP_FUNC: {
        return exp(x);
      }
      case INV_FUNC: {
        return float(1.0 / x);
      }
      case INV_BACK_FUNC: {
        return -(1.0 / (x * x)) * y;
      }
      case IS_CLOSE_FUNC: {
        return (x - y < 1e-2) && (y - x < 1e-2);
      }
      case MAX_FUNC: {
        if (x > y) {
          return x;
        }
        else {
          return y;
        }
      }
      case POW: {
        return pow(x, y);
      }
      case TANH: {
        return tanh(x);
      }
      default: {
        return x + y;
      }
    }
    
}


__device__ int index_to_position(const int* index, const int* strides, int num_dims) {
    int position = 0;
    for (int i = 0; i < num_dims; ++i) {
        position += index[i] * strides[i];
    }
    return position;
}

__device__ void to_index(int ordinal, const int* shape, int* out_index, int num_dims) {
    int cur_ord = ordinal;
    for (int i = num_dims - 1; i >= 0; --i) {
        int sh = shape[i];
        out_index[i] = cur_ord % sh;
        cur_ord /= sh;
    }
}

__device__ void broadcast_index(const int* big_index, const int* big_shape, const int* shape, int* out_index, int num_dims_big, int num_dims) {
    for (int i = 0; i < num_dims; ++i) {
        if (shape[i] > 1) {
            out_index[i] = big_index[i + (num_dims_big - num_dims)];
        } else {
            out_index[i] = 0;
        }
    }
} 


__global__ void MatrixMultiplyKernel(
  float *out,
  const int *out_shape, // [batch_size, m, p]
  const int *out_strides,
  float *a_storage, 
  const int *a_shape, // [batch_size, m, n]
  const int *a_strides,
  float *b_storage,
  const int *b_shape, // [batch_size, n, p]
  const int *b_strides)
{
  /**
  * Multiply two (compact) matrices into an output (also comapct) matrix. Matrix a and b are both in a batch
  * format, with shape [batch_size, m, n], [batch_size, n, p].
  * Requirements:
  * - All data must be first moved to shared memory.
  * - Only read each cell in a and b once.
  * - Only write to global memory once per kernel.
  * There is guarantee that a_shape[0] == b_shape[0], a_shape[2] == b_shape[1],
  * and out_shape[0] == a_shape[0], out_shape[1] == a_shape[1], out_shape[2] == b_shape[2].
  *
  * Args:
  *   out: compact 1D array of size batch_size x m x p to write the output to
  *   out_shape: shape of the output array
  *   out_strides: strides of the output array
  *   a_storage: compact 1D array of size batch_size x m x n
  *   a_shape: shape of the a array
  *   a_strides: strides of the a array
  *   b_storage: compact 1D array of size batch_size x n x p
  *   b_shape: shape of the b array
  *   b_strides: strides of the b array
  *
  * Returns:
  *   None (Fills in out array)
  */

  __shared__ float a_shared[TILE][TILE];
  __shared__ float b_shared[TILE][TILE];

  // In each block, we will compute a batch of the output matrix
  // All the threads in the block will work together to compute this batch
  int batch = blockIdx.z;
  int a_batch_stride = a_shape[0] > 1 ? a_strides[0] : 0;
  int b_batch_stride = b_shape[0] > 1 ? b_strides[0] : 0;

  /// BEGIN HW1_4
  /// TODO
  // Hints:
  // 1. Compute the row and column of the output matrix this block will compute
  // 2. Compute the position in the output array that this thread will write to
  // 3. Iterate over tiles of the two input matrices, read the data into shared memory
  // 4. Synchronize to make sure the data is available to all threads
  // 5. Compute the output tile for this thread block
  // 6. Synchronize to make sure all threads are done computing the output tile for (row, col)
  // 7. Write the output to global memory

  // ===== basic version =====
  // int row = blockIdx.x * blockDim.x + threadIdx.x;
  // int col = blockIdx.y * blockDim.y + threadIdx.y;

  // float sum = 0.0;
  // if (row < out_shape[1] && col < out_shape[2]) {
  //   for (int k = 0; k < a_shape[2]; k++) {
  //     // ex: a_shape = [2, 3, 5] => a_strides = [15, 5, 1]
  //     // if the curr index is (batch, row, k) = (1, 2, 3),
  //     // then a_pos is 1*15 + 2*5 + 3*1.
  //     // same as `index_to_position()`, but just written out manually
  //     int a_pos = batch * a_batch_stride + row * a_strides[1] + k * a_strides[2];
  //     int b_pos = batch * b_batch_stride + k * b_strides[1] + col * b_strides[2];

  //     sum += a_storage[a_pos] * b_storage[b_pos];
  //   }

  //   int out_pos = batch * out_strides[0] + row * out_strides[1] + col * out_strides[2];
  //   out[out_pos] = sum;
  // }

  // ===== optimized version (Tiling)=====
  // (this assumes block size and tile size are the same, both are 32x32)
  // Here,
  // - block size is the number/layout of threads in the CUDA block
  // - tile size is the number/layout of output elements that block computes

  // the position of the output element in the output array that this thread should compute
  int row = blockIdx.x * TILE + threadIdx.x;
  int col = blockIdx.y * TILE + threadIdx.y;

  float p_value = 0;
  for (int phase = 0; phase < (a_shape[2] + TILE - 1) / TILE; phase++) {
    // Load the corresponding block of A and B into shared memory

    // fill in default value first
    a_shared[threadIdx.x][threadIdx.y] = 0;
    b_shared[threadIdx.x][threadIdx.y] = 0;
    
    // if within boundary, then the thread should load the corresponding element from global mem to shared mem
    if (row < a_shape[1] && (phase * TILE + threadIdx.y) < a_shape[2]) {
      int a_pos = batch * a_batch_stride + row * a_strides[1] + (phase * TILE + threadIdx.y) * a_strides[2];
      a_shared[threadIdx.x][threadIdx.y] = a_storage[a_pos];
    }
    if ((phase * TILE + threadIdx.x) < b_shape[1] && col < b_shape[2]) {
      int b_pos = batch * b_batch_stride + (phase * TILE + threadIdx.x) * b_strides[1] + col * b_strides[2];
      b_shared[threadIdx.x][threadIdx.y] = b_storage[b_pos];
    }

    // need to wait for all threads to load the elements from A & B into shared memory
    __syncthreads();

    // multiply the tiles from A & B for the current phase
    for (int k = 0; k < TILE; k++) {
      p_value += a_shared[threadIdx.x][k] * b_shared[k][threadIdx.y];
    }

    // Need to wait for all threads to finish the current phase.
    // Otherwise, if some threads move to the next phase first, then they may overwrite
    // the data in the shared memory that is still currently used by other threads in the current phase
    __syncthreads();
  }

  if (row < a_shape[1] && col < b_shape[2]) {
    int out_pos = batch * out_strides[0] + row * out_strides[1] + col * out_strides[2];
    out[out_pos] = p_value;
  }

  /// END HW1_4
}


__global__ void mapKernel(
  float *out,
  int *out_shape,
  int *out_strides,
  int out_size,
  float *in_storage,
  int *in_shape,
  int *in_strides,
  int shape_size,
  int fn_id)
{
  /**
  * Map function. Apply a unary function to each element of the input array and store the result in the output array.
  * Optimization: Parallelize over the elements of the output array.
  *
  * You may find the following functions useful:
  * - index_to_position: converts an index to a position in a compact array
  * - to_index: converts a position to an index in a multidimensional array
  * - broadcast_index: converts an index in a smaller array to an index in a larger array
  *
  * Args:
  *  out: compact 1D array of size out_size to write the output to
  *  out_shape: shape of the output array
  *  out_strides: strides of the output array
  *  out_size: size of the output array
  *  in_storage: compact 1D array of size in_size
  *  in_shape: shape of the input array
  *  in_strides: strides of the input array
  *  shape_size: number of dimensions in the input and output arrays, assume dimensions are the same
  *  fn_id: id of the function to apply to each element of the input array
  *
  * Returns:
  *  None (Fills in out array)
  */

  int out_index[MAX_DIMS];
  int in_index[MAX_DIMS];

  /// BEGIN HW1_1
  /// TODO
  // Hints:
  // 1. Compute the position in the output array that this thread will write to
  // 2. Convert the position to the out_index according to out_shape
  // 3. Broadcast the out_index to the in_index according to in_shape (optional in some cases)
  // 4. Calculate the position of element in in_array according to in_index and in_strides
  // 5. Calculate the position of element in out_array according to out_index and out_strides
  // 6. Apply the unary function to the input element and write the output to the out memory

  // Definition:
  // - CUDA kernel: a function that runs on the GPU
  // - hierarchy:
  //   - Grid: a grid is all blocks launched for one kernel call
  //   - Block: a block is a group of threads
  //   - Thread: a thread is the thing that does actual work (in sequential order)
  //
  // The below variables are predefined in a kernel function:
  // - blockIdx: the index of the block inside the grid
  // - blockDim: the number of threads inside a block
  // - threadIdx: the index of the thread inside a block

  int out_pos = blockIdx.x * blockDim.x + threadIdx.x; // global thread ID, ex: 7
  if (out_pos >= out_size) {
    return;
  }

  // ex: out_shape = [2, 2, 3]
  // from out_pos = 7, we get out_index = [1, 0, 1]
  to_index(out_pos, out_shape, out_index, shape_size);

  // ex: out_shape = [2, 2, 3], in_shape = [1, 3]
  // Broadcasting rule for two tensors `x` & `y`:
  // 1. align from the right
  // 2. if number of dimensions are not equal, then prepend 1 to the tensor with fewer dimensions to make them equal
  // 3. then, for each dimension size, the resulting dimension size is max(x, y) along that dimension (note: one of them must be 1)
  // => out_shape = [2, 2, 3]
  //    in_shape  = [1, 1, 3]
  broadcast_index(out_index, out_shape, in_shape, in_index, shape_size, shape_size);

  // ex: 
  // in_index = [0, 2]
  // in_strides = [3, 1]
  // => in_pos = 0*3 + 2*1 = 2
  int in_pos = index_to_position(in_index, in_strides, shape_size);
  int out_storage_pos = index_to_position(out_index, out_strides, shape_size);

  out[out_storage_pos] = fn(fn_id, in_storage[in_pos]);

  /// END HW1_1
}


__global__ void reduceKernel(
  float *out,
  int *out_shape,
  int *out_strides,
  int out_size,
  float *a_storage,
  int *a_shape,
  int *a_strides,
  int reduce_dim,
  float reduce_value,
  int shape_size,
  int fn_id)
{
  /**
  * Reduce function. Apply a reduce function to elements of the input array a and store the result in the output array.
  * Optimization:
  * Parallelize over the reduction operation. Each kernel performs one reduction.
  * e.g. a = [[1, 2, 3], [4, 5, 6]], kernel0 computes reduce([1, 2, 3]), kernel1 computes reduce([4, 5, 6]).
  *
  * You may find the following functions useful:
  * - index_to_position: converts an index to a position in a compact array
  * - to_index: converts a position to an index in a multidimensional array
  *
  * Args:
  *  out: compact 1D array of size out_size to write the output to
  *  out_shape: shape of the output array
  *  out_strides: strides of the output array
  *  out_size: size of the output array
  *  a_storage: compact 1D array of size in_size
  *  a_shape: shape of the input array
  *  a_strides: strides of the input array
  *  reduce_dim: dimension to reduce on
  *  reduce_value: initial value for the reduction
  *  shape_size: number of dimensions in the input & output array, assert dimensions are the same
  *  fn_id: id of the reduce function, currently only support add, multiply, and max
  *
  *
  * Returns:
  *  None (Fills in out array)
  */

  // Shared memory is a memory space that can be accessed by all threads in a thread block.
  // It is located on each SM.
  __shared__ float cache[BLOCK_DIM]; // Uncomment this line if you want to use shared memory to store partial results
  int out_index[MAX_DIMS];
  int a_index[MAX_DIMS];

  /// BEGIN HW1_3
  /// TODO
  // 1. Define the position of the output element that this thread or this block will write to
  // 2. Convert the out_pos to the out_index according to out_shape
  // 3. Initialize the reduce_value to the output element
  // 4. Iterate over the reduce_dim dimension of the input array to compute the reduced value
  // 5. Write the reduced value to out memory

  // ===== Basic version =====
  // 1 thread is responsible for 1 output in the output tensor
  //
  // int out_pos = blockIdx.x * blockDim.x + threadIdx.x; // global thread ID, ex: 7
  // if (out_pos >= out_size) {
  //   return;
  // }

  // to_index(out_pos, out_shape, out_index, shape_size);

  // int out_storage_pos = index_to_position(out_index, out_strides, shape_size);

  // float result = reduce_value;
  // for (int i = 0; i < a_shape[reduce_dim]; i++) {
  //   out_index[reduce_dim] = i;

  //   int a_pos = index_to_position(out_index, a_strides, shape_size);

  //   result = fn(fn_id, result, a_storage[a_pos]);
  // }

  // out[out_storage_pos] = result;

  // ===== Optimized version =====
  //  1 block is responsible for 1 output in the output tensor
  // ex: input_tensor's shape = [2, 3, 4]
  // suppose a block has 4 threads. For output position [0, 1, 0], block1 will do
  // - thread0 loads a[0, 1, 0] into shared memory
  // - thread1 loads a[0, 1, 1] ...
  // - thread2 loads a[0, 1, 2] ...
  // - thread3 loads a[0, 1, 3] ...
  // so now cache is [v0, v1, v2, v3].
  // Next, reduce in parallel
  // round1: thread0 combines v0 + v1; thread2 combines v2 + v3 => cache = [v0+v1, _, v2+v3, _]
  // round2: thread0 combines v0+v1 + v2+v3 => cache[0] = total

  int out_pos = blockIdx.x;
  int tid = threadIdx.x; // the index of the element inside the reduction
  if (out_pos >= out_size) {
    return;
  }

  to_index(out_pos, out_shape, out_index, shape_size); // the index of the output element this block is computing

  // start from this output index
  for (int d = 0; d < shape_size; d++) {
    a_index[d] = out_index[d];
  }

  // the kernel is launched with threadsPerBlock = 32.
  // However, the # of elements along reduce_dim may be > 32.
  // Thus, we need to add a preprocessing step where each thread sums up a[, , start_idx], a[, , start_idx + threadsPerBlock], ... and store it in a[, , 0].
  // ex: thread0 sums a[, , 0], a[, , 32], a[, , 64] and store the sum in a[, , 0]
  //     thread1 sums a[, , 1], a[, , 33], a[, , 65] and store the sum in a[, , 1]
  // Now, we preprocess the array as if it had <= 32 elements along reduce_dim
  int reduce_size = a_shape[reduce_dim]; // # of elements along reduce_dim
  int tree_size = min(reduce_size, (int)blockDim.x); // # of elements along reduce_dim after the preprocessing
  // let each thread handle one element in preprocessed array
  if (tid < tree_size) {
    a_index[reduce_dim] = tid;
    int a_pos = index_to_position(a_index, a_strides, shape_size);
    float local = a_storage[a_pos]; // ex: a[, , 0]
    for (int i = tid + blockDim.x; i < reduce_size; i += blockDim.x) {
      a_index[reduce_dim] = i;
      a_pos = index_to_position(a_index, a_strides, shape_size);
      local = fn(fn_id, local, a_storage[a_pos]);
    }

    cache[tid] = local;
  }

  // need to wait for all threads to finish, bcuz we want to ensure
  // all the elements along preprocessed reduce_dim to be loaded into the shared memory
  __syncthreads();

  // tree reduction
  // the thread IDs that will do the work in each round is
  // round1: t0, t2, t4, ...
  // round2: t0, t4, t8, ...
  // in total there is log2(reduce_size) rounds
  for (int offset = 1; offset < tree_size; offset *= 2) {
    if (tid % (2 * offset) == 0 && tid + offset < tree_size) {
      // thread IDs that do the work in the current round
      cache[tid] = fn(fn_id, cache[tid], cache[tid + offset]);
    }

    __syncthreads(); // wait for all threads in the current round to finish
  }

  // first thread writes the final result
  if (tid == 0) {
    int out_storage_pos = index_to_position(out_index, out_strides, shape_size);
    out[out_storage_pos] = fn(fn_id, reduce_value, cache[0]); // final reduce with the initial reduce value
  }


  /// END HW1_3
}

__global__ void zipKernel(
  float *out,
  int *out_shape,
  int *out_strides,
  int out_size,
  int out_shape_size,
  float *a_storage,
  int *a_shape,
  int *a_strides,
  int a_shape_size,
  float *b_storage,
  int *b_shape,
  int *b_strides,
  int b_shape_size,
  int fn_id)
{
  /**
  * Zip function. Apply a binary function to elements of the input array a & b and store the result in the output array.
  * Optimization: Parallelize over the elements of the output array.
  *
  * You may find the following functions useful:
  * - index_to_position: converts an index to a position in a compact array
  * - to_index: converts a position to an index in a multidimensional array
  * - broadcast_index: converts an index in a smaller array to an index in a larger array
  *
  * Args:
  *  out: compact 1D array of size out_size to write the output to
  *  out_shape: shape of the output array
  *  out_strides: strides of the output array
  *  out_size: size of the output array
  *  out_shape_size: number of dimensions in the output array
  *  a_storage: compact 1D array of size in_size
  *  a_shape: shape of the input array
  *  a_strides: strides of the input array
  *  a_shape_size: number of dimensions in the input array
  *  b_storage: compact 1D array of size in_size
  *  b_shape: shape of the input array
  *  b_strides: strides of the input array
  *  b_shape_size: number of dimensions in the input array
  *  fn_id: id of the function to apply to each element of the a & b array
  *
  *
  * Returns:
  *  None (Fills in out array)
  */

  int out_index[MAX_DIMS];
  int a_index[MAX_DIMS];
  int b_index[MAX_DIMS];

  /// BEGIN HW1_2
  /// TODO
  // Hints:
  // 1. Compute the position in the output array that this thread will write to
  // 2. Convert the position to the out_index according to out_shape
  // 3. Calculate the position of element in out_array according to out_index and out_strides
  // 4. Broadcast the out_index to the a_index according to a_shape
  // 5. Calculate the position of element in a_array according to a_index and a_strides
  // 6. Broadcast the out_index to the b_index according to b_shape
  // 7.Calculate the position of element in b_array according to b_index and b_strides
  // 8. Apply the binary function to the input elements in a_array & b_array and write the output to the out memory

  int out_pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (out_pos >= out_size) {
    return;
  }

  to_index(out_pos, out_shape, out_index, out_shape_size);

  int out_storage_pos = index_to_position(out_index, out_strides, out_shape_size);

  // out_index -> find corresponding input_index (a & b) under broadcasting
  broadcast_index(out_index, out_shape, a_shape, a_index, out_shape_size, a_shape_size);
  int a_pos = index_to_position(a_index, a_strides, a_shape_size);

  broadcast_index(out_index, out_shape, b_shape, b_index, out_shape_size, b_shape_size);
  int b_pos = index_to_position(b_index, b_strides, b_shape_size);

  out[out_storage_pos] = fn(fn_id, a_storage[a_pos], b_storage[b_pos]);

  /// END HW1_2
}


extern "C" {

void MatrixMultiply(
    float* out,
    int* out_shape,
    int* out_strides,
    float* a_storage,
    int* a_shape,
    int* a_strides,
    float* b_storage,
    int* b_shape,
    int* b_strides,
    int batch, int m, int p
) {
    int n = a_shape[2];

    // Allocate device memory
    float *d_out, *d_a, *d_b;
    cudaMalloc(&d_a, batch * m * n * sizeof(float));
    cudaMalloc(&d_b, batch * n * p * sizeof(float));
    cudaMalloc(&d_out, batch * m * p * sizeof(float));

    int *d_out_shape, *d_out_strides, *d_a_shape, *d_a_strides, *d_b_shape, *d_b_strides;
    cudaMalloc(&d_out_shape, 3 * sizeof(int));
    cudaMalloc(&d_out_strides, 3 * sizeof(int));
    cudaMalloc(&d_a_shape, 3 * sizeof(int));
    cudaMalloc(&d_a_strides, 3 * sizeof(int));
    cudaMalloc(&d_b_shape, 3 * sizeof(int));
    cudaMalloc(&d_b_strides, 3 * sizeof(int));


    // Copy data to the device
    cudaMemcpy(d_a, a_storage, batch * m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_storage, batch * n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_shape, a_shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_strides, a_strides, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_shape, b_shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_strides, b_strides, 3 * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = BASE_THREAD_NUM;
    dim3 blockDims(threadsPerBlock, threadsPerBlock, 1); // Adjust these values based on your specific requirements
    dim3 gridDims((m + threadsPerBlock - 1) / threadsPerBlock, (p + threadsPerBlock - 1) / threadsPerBlock, batch);
    MatrixMultiplyKernel<<<gridDims, blockDims>>>(
        d_out, d_out_shape, d_out_strides, d_a, d_a_shape, d_a_strides, d_b, d_b_shape, d_b_strides
    );

    // Copy back to the host
    cudaMemcpy(out, d_out, batch * m * p * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();

    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Matmul Error: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_a_shape);
    cudaFree(d_a_strides);
    cudaFree(d_b_shape);
    cudaFree(d_b_strides);
}

void tensorMap(
    float* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size, 
    float* in_storage, 
    int* in_shape, 
    int* in_strides,
    int in_size,
    int shape_size,
    int fn_id
) {

    float *d_out, *d_in;
    cudaMalloc(&d_out, out_size * sizeof(float));
    cudaMalloc(&d_in, in_size * sizeof(float));

    int *d_out_shape, *d_out_strides, *d_in_shape, *d_in_strides;
    cudaMalloc(&d_out_shape, shape_size * sizeof(int));
    cudaMalloc(&d_out_strides, shape_size * sizeof(int));
    cudaMalloc(&d_in_shape, shape_size * sizeof(int));
    cudaMalloc(&d_in_strides, shape_size * sizeof(int));

    cudaMemcpy(d_in, in_storage, in_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_shape, in_shape, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_strides, in_strides, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = BASE_THREAD_NUM;
    int blocksPerGrid = (out_size + threadsPerBlock - 1) / threadsPerBlock;
    mapKernel<<<blocksPerGrid, threadsPerBlock>>>(
      d_out, d_out_shape, d_out_strides, out_size, 
      d_in, d_in_shape, d_in_strides, 
      shape_size, fn_id);
    
    // Copy back to the host
    cudaMemcpy(out, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Map Error: %s\n", cudaGetErrorString(err));
      // Handle the error (e.g., by exiting the program)
      exit(EXIT_FAILURE);
    }

    // Free memory on device
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_in_shape);
    cudaFree(d_in_strides);
}


void tensorZip(
    float* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size,
    int out_shape_size,
    float* a_storage, 
    int* a_shape, 
    int* a_strides,
    int a_size,
    int a_shape_size,
    float* b_storage, 
    int* b_shape, 
    int* b_strides,
    int b_size,
    int b_shape_size,
    int fn_id
) {

    // Allocate device memory
    float *d_out, *d_a, *d_b;
    cudaMalloc((void **)&d_a, a_size * sizeof(float));
    cudaMalloc(&d_b, b_size * sizeof(float));
    cudaMalloc(&d_out, out_size * sizeof(float));

    int *d_out_shape, *d_out_strides, *d_a_shape, *d_a_strides, *d_b_shape, *d_b_strides;
    cudaMalloc(&d_out_shape, out_shape_size * sizeof(int));
    cudaMalloc(&d_out_strides, out_shape_size * sizeof(int));
    cudaMalloc(&d_a_shape, a_shape_size * sizeof(int));
    cudaMalloc(&d_a_strides, a_shape_size * sizeof(int));
    cudaMalloc(&d_b_shape, b_shape_size * sizeof(int));
    cudaMalloc(&d_b_strides, b_shape_size * sizeof(int));

    // Copy data to the device
    cudaMemcpy(d_a, a_storage, a_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_storage, b_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, out_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, out_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_shape, a_shape, a_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_strides, a_strides, a_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_shape, b_shape, b_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_strides, b_strides, b_shape_size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = BASE_THREAD_NUM;
    int blocksPerGrid = (out_size + threadsPerBlock - 1) / threadsPerBlock;
    zipKernel<<<blocksPerGrid, threadsPerBlock>>>(
      d_out, d_out_shape, d_out_strides, out_size, out_shape_size,
      d_a, d_a_shape, d_a_strides, a_shape_size,
      d_b, d_b_shape, d_b_strides, b_shape_size,
      fn_id);

    // Copy back to the host
    cudaMemcpy(out, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();


    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Zip Error: %s\n", cudaGetErrorString(err));
      // Handle the error (e.g., by exiting the program)
      exit(EXIT_FAILURE);
    }

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_a_shape);
    cudaFree(d_a_strides);
    cudaFree(d_b_shape);
    cudaFree(d_b_strides);
}



void tensorReduce(
    float* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size, 
    float* a_storage, 
    int* a_shape, 
    int* a_strides, 
    int reduce_dim, 
    float reduce_value,
    int shape_size,
    int fn_id
) {
    int a_size = out_size * a_shape[reduce_dim];
    float *d_out, *d_a;
    cudaMalloc(&d_out, out_size * sizeof(float));
    cudaMalloc(&d_a, a_size * sizeof(float));

    int *d_out_shape, *d_out_strides, *d_a_shape, *d_a_strides;
    cudaMalloc(&d_out_shape, shape_size * sizeof(int));
    cudaMalloc(&d_out_strides, shape_size * sizeof(int));
    cudaMalloc(&d_a_shape, shape_size * sizeof(int));
    cudaMalloc(&d_a_strides, shape_size * sizeof(int));

    cudaMemcpy(d_a, a_storage, a_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_shape, a_shape, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_strides, a_strides, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = BASE_THREAD_NUM;
    int blocksPerGrid = (out_size + threadsPerBlock - 1) / threadsPerBlock;
    // reduceKernel<<<blocksPerGrid, threadsPerBlock>>>(
    //     d_out, d_out_shape, d_out_strides, out_size, 
    //     d_a, d_a_shape, d_a_strides, 
    //     reduce_dim, reduce_value, shape_size, fn_id
    // );

    reduceKernel<<<out_size, threadsPerBlock>>>(
      d_out, d_out_shape, d_out_strides, out_size,
      d_a, d_a_shape, d_a_strides,
      reduce_dim, reduce_value, shape_size, fn_id);

    // Copy back to the host
    cudaMemcpy(out, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();

    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Reduce Error: %s\n", cudaGetErrorString(err));
      // Handle the error (e.g., by exiting the program)
      exit(EXIT_FAILURE);
    }

    cudaFree(d_a);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_a_shape);
    cudaFree(d_a_strides);
}

}