#include "includes/block_reduce.h"
#include "includes/kernels.h"
#include "includes/cuda_util.h"

#include <cooperative_groups.h>
#include <cstddef>

namespace cg = cooperative_groups;
namespace lightseq {
namespace cuda {

const float LN_EPSILON = 1e-8f;
#define TILE_DIM 32


/**
  @brief: ker_layer_norm
  Standard layer normalization.
  It will not only output the layer norm result,
    but also outputs variance.
    may also output means, depends on whether
    the means argument is nullptr

  @thread
  gridDim.x = batch_size * seq_len
  blockDim.x = hidden_size

  @param ln_res: [batch_size * seq_len, hidden_size], ln result.
  @param vars: [batch_size * seq_len], variance per token
  @param means: [batch_size * seq_len], means per token, can be nullptr
  @param inp: [batch_size * seq_len, hidden_size], ln input.
  @param scale: [hidden_size], ln scale
  @param bias: [hidden_size], ln bias
  */
template <typename T>
__global__ void ker_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                               const T *scale, const T *bias, int hidden_size) {
  // Launch configuration:
  // one block handles one token
  /// BEGIN ASSIGN4_2_1
  /// TODO
  // Hints:
  // 1. Compute x and x^2 with reinterpret_cast by casting to float4 for speedup
  // 2. Compute reduce sum with blockReduce and add epsilon with LN_EPSILON
  // 3. Compute layernorm result with reinterpret_cast by casting to float4 for speedup
  
  // Step 1: compute sum and square_sum
  float l_sum = 0;
  float l_square_sum = 0;
  // (B, T, C) is flattened to 2D (B*T, C)
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size; // move inp to the starting row that corresponds to the token this block will handle
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    l_sum += val.x + val.y + val.z + val.w;
    l_square_sum += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
  }

  // Step 2: compute mean and var
  float reduce_vals[2] = {l_sum, l_square_sum};
  // use blockReduce because each block may have > 32 threads (> 1 warp)
  blockReduce<ReduceType::kSum, 2>(reduce_vals);

  __shared__ float s_mean;
  __shared__ float s_var;
  if (threadIdx.x == 0) {
    float embed_dim = hidden_size * 4.0f;
    s_mean = reduce_vals[0] / embed_dim;
    s_var = reduce_vals[1] / embed_dim - s_mean * s_mean + LN_EPSILON; // var(x) = E[X^2] - E[X]^2
    means[blockIdx.x] = s_mean;
    vars[blockIdx.x] = s_var;
  }

  __syncthreads();

  // Step 3: normalize each token
  float mean = s_mean;
  float inverse_std = rsqrtf(s_var);

  float4 *output_f4 = reinterpret_cast<float4 *>(ln_res) + blockIdx.x * hidden_size;
  const float4 *scale_f4 = reinterpret_cast<const float4 *>(scale);
  const float4 *bias_f4 = reinterpret_cast<const float4 *>(bias);
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    float4 gamma = scale_f4[idx];
    float4 beta = bias_f4[idx];

    val.x = (val.x - mean) * inverse_std * gamma.x + beta.x;
    val.y = (val.y - mean) * inverse_std * gamma.y + beta.y;
    val.z = (val.z - mean) * inverse_std * gamma.z + beta.z;
    val.w = (val.w - mean) * inverse_std * gamma.w + beta.w;
    output_f4[idx] = val;
  }
  /// END ASSIGN4_2_1
}

extern "C" {
void launch_layernorm(float *ln_res, float *vars, float *means,
                              const float *inp, const float *scale,
                              const float *bias, int batch_size, int hidden_dim,
                              cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  int float_size = sizeof(float);
  int input_size = batch_size * hidden_dim * float_size;
  int scale_size = hidden_dim * float_size;
  int bias_size = hidden_dim * float_size;
  int output_size = batch_size * hidden_dim * float_size;
  int mean_size = batch_size * float_size;
  int var_size = batch_size * float_size;


  float *d_ln_res, *d_vars, *d_means, *d_inp, *d_scale, *d_bias;
  cudaMalloc((void **)&d_ln_res, output_size);
  cudaMalloc((void **)&d_vars, var_size);
  cudaMalloc((void **)&d_means, mean_size);
  cudaMalloc((void **)&d_inp, input_size);
  cudaMalloc((void **)&d_scale, scale_size);
  cudaMalloc((void **)&d_bias, bias_size);

  cudaMemcpy(d_inp, inp, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scale, scale, scale_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias, bias_size, cudaMemcpyHostToDevice);

  // For using float4
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<float><<<grid_dim, block_dim, 0, stream>>>(
      d_ln_res, d_vars, d_means, d_inp, d_scale, d_bias, hidden_dim);

  // Copy back to the host
  cudaMemcpy(ln_res, d_ln_res, output_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(vars, d_vars, var_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(means, d_means, mean_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Check CUDA execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm Error: %s\n", cudaGetErrorString(err));
    // Handle the error (e.g., by exiting the program)
    exit(EXIT_FAILURE);
  }

  // Free memory on device
  cudaFree(d_ln_res);
  cudaFree(d_vars);
  cudaFree(d_means);
  cudaFree(d_inp);
  cudaFree(d_scale);
  cudaFree(d_bias);

}
}

/**
  @brief: ker_ln_bw_dgamma_dbetta
  Layer norm backward kernel, compute the gradient of gamma and betta.
  dbetta = sum(dout, dim=0)
  dgamma = sum(xhat * dout, dim=0)
  xhat = (input - mean) * rsqrt(var) or
    (output - betta) / gamma

  @thread
  gridDim.x = hidden_size / 32
  blockDim.x = 32
  blockDim.y = 32

  @param gamma_grad: [hidden_size], gradient of gamma
  @param betta_grad: [hidden_size], gradient of betta
  @param out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
  @param inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
    ln input if means is not nullptr
  @param gamma: [hidden_size], gamma of ln,
    used to compute xhat, maybe nullptr
  @param betta: [hidden_size], betta of ln,
    used to compute xhat, maybe nullptr
  @param vars: [batch_size * seq_len], variance of ln forward,
    used to compute xhat, maybe nullptr
  @param means: [batch_size * seq_len], mean of ln forward,
    used to compute xhat, maybe nullptr
  (gamma && betta) ^ (vars && means) should be true
  @param rows: (batch_size * seq_len,)
  @param width: (hidden_dim,)
  */
template <typename T>
__global__ void ker_ln_bw_dgamma_dbetta(T *gamma_grad, T *betta_grad,
                                        const T *out_grad,
                                        const T *inp, const T *gamma,
                                        const T *betta, const T *vars,
                                        const T *means, int rows, int width) {

  /// BEGIN ASSIGN4_2_2
  /// TODO
  // Hints:
  // 1. Compute the partial gradients by looping across inp rows
  // 2. Store the partial gradients in the shared memory arrays
  // 3. Compute the reduce sum of the shared memory arrays with g.shfl_down
  //      -> More hints about `g.shfl_down`:
  //      -> https://developer.nvidia.com/blog/cooperative-groups/#:~:text=Using%20thread_block_tile%3A%3Ashfl_down()%20to%20simplify%20our%20warp%2Dlevel%20reduction%20does%20benefit%20our%20code%3A%20it%20simplifies%20it%20and%20eliminates%20the%20need%20for%20shared%20memory
  //      -> The highlighted line gives you a conceptual understanding of what the g.shfl_down is doing. Usually, the threads inside a block need to load everything to shared memory and work together to reduce the result (like what you have implemented in the hw1 for reduce function). 
  //      -> Now g.shfl_down helps you do so without consuming any shared memory. g.shfl_down makes it more efficient.
  // 4. Assign the final result to the correct position in the global output

  __shared__ float betta_buffer[TILE_DIM][TILE_DIM];
  __shared__ float gamma_buffer[TILE_DIM][TILE_DIM];

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

  // Step 1: compute d_beta & d_gamma
  int col = blockIdx.x * TILE_DIM + threadIdx.x;

  float d_beta = 0.0f; // partial sum for column `col` using rows `threadIdx.y`, `threadIdx.y + 32`, ...
  float d_gamma = 0.0f;
  if (col < width) {
    for (int row = threadIdx.y; row < rows; row += TILE_DIM) { // tile-wise padding in the row dimension
      int idx = row * width + col;

      float dy = (float)out_grad[idx];
      float xhat;
      if (means != nullptr) {
        xhat = ((float)inp[idx] - (float)means[row]) * rsqrtf((float)vars[row]);
      } else { // y = gamma * x_hat + beta => x_hat = (y - beta) / gamma
        xhat = ((float)inp[idx] - (float)betta[col]) / (float)gamma[col];
      }

      d_beta += dy;
      d_gamma += dy * xhat;
    }
  }

  // Step 2: store the partial gradients into shared memory
  // transpose the index to (x, y) so that the partial sum of column `col` from (row=0,col), (row=1,col), ..., (row=31,col) is stored as a single row instead of a column
  //
  // Note: CUDA fills a warp by increasing `x` first, then `y`, then `z`, where every 32 threads form a warp.
  // Since in the launch config, we define the dimension `x` as the column, and the dimension `y` as the row,
  // so in this case, one warp correspond to one horizontal row of 32 columns.
  //
  // The reason to transpose here is bcuz warp shuffle only works between threads in the same warp,
  // and since in this case one warp correspond to one horizontal row, so we need to store the 
  // 32 partial sum of d_beta_col as a row instead of a column.
  betta_buffer[threadIdx.x][threadIdx.y] = d_beta;
  gamma_buffer[threadIdx.x][threadIdx.y] = d_gamma;

  __syncthreads();
  
  // Step 3: reduce the partial sum using g.shfl_down()
  float d_beta_sum = betta_buffer[threadIdx.y][threadIdx.x]; // one warp read one row: all 32 partial sum of d_beta_col
  float d_gamma_sum = gamma_buffer[threadIdx.y][threadIdx.x];
  for (int offset = TILE_DIM / 2; offset > 0; offset >>= 1) {
    d_beta_sum += g.shfl_down(d_beta_sum, offset);
    d_gamma_sum += g.shfl_down(d_gamma_sum, offset);
  }
  // shfl_down vs shfl_xor:
  // 1. shfl_down: sends value from higher lane (thread) to lower lane. In the end only thread 0 gets the full reduced val.
  // ex: start = [a, b, c, d]
  //  offset = 2 [a+c, b+d, c, d]
  //  offset = 1 [a+b+c+d, b+c+d, c+d, d]
  // 2. shfl_xor: exchange values with other lanes. In the end all threads in the warp get the full reduced val.
  // ex: start = [a, b, c, d]
  //  offset = 2 [a+c, b+d, c+a, d+b]
  //  offset = 1 [a+b+c+d, a+b+c+d, a+b+c+d, a+b+c+d]
  
  // Step 4: write the reduced result back to global mem
  // after g.shfl_down, only the first thread in each warp gets the full reduced value
  if (threadIdx.x == 0) {
    int out_col = blockIdx.x * TILE_DIM + threadIdx.y; // use threadIdx.y here, as it represents which column this warp reduced (d_beta_col)
    if (out_col < width) {
      betta_grad[out_col] = d_beta_sum;
      gamma_grad[out_col] = d_gamma_sum;
    }
  }

  /// END ASSIGN4_2_2
}

/**
  @brief: ker_ln_bw_dinp
  Layer norm backward kernel, compute the gradient of input.
  dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden_dim)
    * rsqrt(var)
  xhat = (input - mean) * rsqrt(var) if mean is not nullptr
        (output - betta) / gamma if mean is nullptr
  dxhat = dout * gamma


  @thread
  gridDim.x = batch_size * seq_len
  blockDim.x = hidden_size

  @param inp_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
  @param out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
  @param residual_grad: [batch_size * seq_len, hidden_size], gradient of residual input,
    usually appear in pre-layer-norm for transformer layer, maybe nullptr
  @param inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
    ln input if means is not nullptr
  @param gamma: [hidden_size], gamma of ln, used to compute xhat and dxhat
  @param betta: [hidden_size], betta of ln, used to compute xhat, maybe nullptr
  @param vars: [batch_size * seq_len], variance of ln forward, used to compute xhat and dinp
  @param means: [batch_size * seq_len], mean of ln forward, used to compute xhat, maybe nullptr
  */
  template <typename T>
__global__ void ker_ln_bw_dinp(T *inp_grad, const T *out_grad, const T *inp,
                               const T *gamma, const T *betta, const T *vars,
                               const T *means, int hidden_dim) {
  
  /// BEGIN ASSIGN4_2_2
  /// TODO
  // Hints:
  // 1. Compute dxhat=dy*w with reinterpret_cast by casting to float4 for speedup
  // 2. Compute xhat with reinterpret_cast by casting to float4 for speedup
  // 3. Compute reduce sum for dxhat and dxhat*xhat with blockReduce
  // 4. Compute final gradient

  // Launch config: one block is responsible for 1 token / row
  
  // Step 1
  const float4 *out_grad_f4 = reinterpret_cast<const float4 *>(out_grad) + blockIdx.x * hidden_dim;
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_dim;
  const float4 *gamma_f4 = reinterpret_cast<const float4 *>(gamma);
  const float4 *betta_f4 = reinterpret_cast<const float4 *>(betta);

  float l_sum = 0.0f; // partial sum of dxhat
  float l_sum_xhat = 0.0f; // partial sum of dxhat * xhat
  for (int idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
    // Step 1: dxhat = dout * gamma
    float4 dy = out_grad_f4[idx];
    float4 g = gamma_f4[idx];

    float4 dxhat;
    dxhat.x = dy.x * g.x;
    dxhat.y = dy.y * g.y;
    dxhat.z = dy.z * g.z;
    dxhat.w = dy.w * g.w;

    // Step 2: compute xhat
    float4 x = inp_f4[idx];
    float4 xhat;

    if (means != nullptr) {
      float mean = means[blockIdx.x];
      float inv_std = rsqrtf(vars[blockIdx.x]);

      xhat.x = (x.x - mean) * inv_std;
      xhat.y = (x.y - mean) * inv_std;
      xhat.z = (x.z - mean) * inv_std;
      xhat.w = (x.w - mean) * inv_std;
    } else {
      float4 b = betta_f4[idx];

      xhat.x = (x.x - b.x) / g.x;
      xhat.y = (x.y - b.y) / g.y;
      xhat.z = (x.z - b.z) / g.z;
      xhat.w = (x.w - b.w) / g.w;
    }

    // partial sums needed for Step 3
    l_sum += dxhat.x + dxhat.y + dxhat.z + dxhat.w;
    l_sum_xhat += dxhat.x * xhat.x + dxhat.y * xhat.y + dxhat.z * xhat.z + dxhat.w * xhat.w;
  }

  // Step 3
  float reduce_vals[2] = {l_sum, l_sum_xhat};
  blockReduce<ReduceType::kSum, 2>(reduce_vals);

  __shared__ float s_sum_dxhat; // full sum of dxhat
  __shared__ float s_sum_dxhat_xhat; // full sum of dxhat * xhat
  if (threadIdx.x == 0) {
      s_sum_dxhat = reduce_vals[0];
      s_sum_dxhat_xhat = reduce_vals[1];
  }

  __syncthreads();
 
  // Step 4
  float inv_std = rsqrtf(vars[blockIdx.x]);
  float H = hidden_dim * 4.0f;

  float4 *inp_grad_f4 = reinterpret_cast<float4 *>(inp_grad) + blockIdx.x * hidden_dim;
  for (int idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
    float4 dy = out_grad_f4[idx];
    float4 g = gamma_f4[idx];
    float4 x = inp_f4[idx];

    float4 dxhat;
    dxhat.x = dy.x * g.x;
    dxhat.y = dy.y * g.y;
    dxhat.z = dy.z * g.z;
    dxhat.w = dy.w * g.w;

    float4 xhat;
    if (means != nullptr) {
      float mean = means[blockIdx.x];

      xhat.x = (x.x - mean) * inv_std;
      xhat.y = (x.y - mean) * inv_std;
      xhat.z = (x.z - mean) * inv_std;
      xhat.w = (x.w - mean) * inv_std;
    } else {
      float4 b = betta_f4[idx];

      xhat.x = (x.x - b.x) / g.x;
      xhat.y = (x.y - b.y) / g.y;
      xhat.z = (x.z - b.z) / g.z;
      xhat.w = (x.w - b.w) / g.w;
    }

    float4 dx;
    dx.x = (dxhat.x - (s_sum_dxhat + xhat.x * s_sum_dxhat_xhat) / H) * inv_std;
    dx.y = (dxhat.y - (s_sum_dxhat + xhat.y * s_sum_dxhat_xhat) / H) * inv_std;
    dx.z = (dxhat.z - (s_sum_dxhat + xhat.z * s_sum_dxhat_xhat) / H) * inv_std;
    dx.w = (dxhat.w - (s_sum_dxhat + xhat.w * s_sum_dxhat_xhat) / H) * inv_std;

    inp_grad_f4[idx] = dx;
  }
  /// END ASSIGN4_2_2
}
extern "C" {
void launch_layernorm_bw(float *gamma_grad, float *betta_grad, float *inp_grad,
                         const float *out_grad, const float *inp, const float *gamma,
                         const float *betta, const float *vars,
                         const float *means, int batch_size, int hidden_dim,
                         cudaStream_t stream_1, cudaStream_t stream_2) {
  
  // Allocate device memory
  float *d_gamma_grad, *d_betta_grad, *d_inp_grad, *d_out_grad, *d_inp, *d_gamma, *d_betta, *d_vars, *d_means;
  int grad_output_size = batch_size * hidden_dim * sizeof(float);
  int gamma_betta_size = hidden_dim * sizeof(float);
  int vars_means_size = batch_size * sizeof(float);

  cudaMalloc((void **)&d_gamma_grad, gamma_betta_size);
  cudaMalloc((void **)&d_betta_grad, gamma_betta_size);
  cudaMalloc((void **)&d_inp_grad, grad_output_size);
  cudaMalloc((void **)&d_out_grad, grad_output_size);
  cudaMalloc((void **)&d_inp, grad_output_size);
  cudaMalloc((void **)&d_gamma, gamma_betta_size);
  cudaMalloc((void **)&d_betta, gamma_betta_size);
  cudaMalloc((void **)&d_vars, vars_means_size);
  cudaMalloc((void **)&d_means, vars_means_size);

  // Copy memory to device
  cudaMemcpy((void *)d_out_grad, out_grad, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_inp, inp, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_gamma, gamma, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_betta, betta, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_vars, vars, vars_means_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_means, means, vars_means_size, cudaMemcpyHostToDevice);

  // Launch kernels
  // Compute grad of gamma and betta
  // This calculates the number of blocks needed to cover the data along the specified dimension, rounds it up.
  dim3 grid_dim((hidden_dim + TILE_DIM - 1) / TILE_DIM);
  dim3 block_dim(TILE_DIM, TILE_DIM);
  ker_ln_bw_dgamma_dbetta<float><<<grid_dim, block_dim, 0, stream_1>>>(
      d_gamma_grad, d_betta_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars,
      d_means, batch_size, hidden_dim);

  // Compute grad of input
  if (hidden_dim % 4 != 0 || hidden_dim > 4096) {
    throw std::runtime_error("hidden_dim % 4 != 0 || hidden_dim > 4096");
  }
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  ker_ln_bw_dinp<<<batch_size, nthread, 0, stream_2>>>(
      d_inp_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars, d_means, hidden_dim);

  // Synchronize and check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm_bw Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy back to host
  cudaMemcpy(gamma_grad, d_gamma_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(betta_grad, d_betta_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(inp_grad, d_inp_grad, grad_output_size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_gamma_grad);
  cudaFree(d_betta_grad);
  cudaFree(d_inp_grad);
  cudaFree((void *)d_out_grad);
  cudaFree((void *)d_inp);
  cudaFree((void *)d_gamma);
  cudaFree((void *)d_betta);
  cudaFree((void *)d_vars);
  cudaFree((void *)d_means);
}}
}}
