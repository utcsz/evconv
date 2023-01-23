#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>

#include "ops.h"
#include "utils.h"
#include "checks.h"

template <typename scalar_t>
__device__ __forceinline__ scalar_t activation(scalar_t x) {
    return x < 0.0f ? 0.0f : x; 
}

// assuems C,H,W format
template <typename scalar_t, int C_PER_BLOCK=3, int H_PER_BLOCK=3, int W_PER_BLOCK=3, int WARP_SIZE=32>
__global__ void activation_increment_kernel(
    scalar_t *__restrict__  X,
    scalar_t const *__restrict__ in_incr,
    scalar_t * __restrict__ out_incr,  // expect a zero tenor, out
    int C, int H, int W
){
    // int const warp_idx = threadIdx.x/WARP_SIZE;
    int const lane_idx = threadIdx.x%WARP_SIZE;
    int const block_idx = blockIdx.x;

    int const H_up = divup(H, H_PER_BLOCK);
    int const W_up = divup(W, W_PER_BLOCK);
    int const HW_up = H_up*W_up;

    int const c_in_start = block_idx/HW_up;
    int const c_in_end = min(C, c_in_start + C_PER_BLOCK);

    int const w = W_PER_BLOCK*(block_idx%W_up) + lane_idx%W_PER_BLOCK;
    int const h = H_PER_BLOCK*(block_idx/W_up) + lane_idx/W_PER_BLOCK;

    // out of bounds
    if(lane_idx >= H_PER_BLOCK*W_PER_BLOCK || h > H || w > W)
        return;

    int const px_offs = h*W + w;

    for(int c = c_in_start; c < c_in_end ; c += 1){
        int x_id = c*H*W + px_offs;
        scalar_t* reserve = &X[x_id];
        scalar_t const * incr = &in_incr[x_id];
        scalar_t const full = *reserve + *incr;
        out_incr[x_id] = activation(full) - activation(*reserve);
        *reserve = full;
    }
}

template <typename scalar_t, int C_PER_BLOCK=16, int H_PER_BLOCK=3, int W_PER_BLOCK=3>
void activation_increment_cuda(
    torch::Tensor &X,
    torch::Tensor const &in_incr,
    torch::Tensor &out_incr  // expect a zero tensor
){
    auto X_dim = dim(X.sizes());

    // per block function: 3*3*C_PER_BLOCK
    int const H_up = divup(X_dim.H, H_PER_BLOCK);
    int const W_up = divup(X_dim.W, W_PER_BLOCK);
    int const C_up = divup(X_dim.C, C_PER_BLOCK);

    int const blocks = H_up*C_up*W_up;
    int const threads = 32;

    activation_increment_kernel<scalar_t, C_PER_BLOCK, H_PER_BLOCK, W_PER_BLOCK><<<blocks, threads>>>(
        X.data_ptr<scalar_t>(), 
        in_incr.data_ptr<scalar_t>(), 
        out_incr.data_ptr<scalar_t>(), 
        X_dim.C, X_dim.H, X_dim.W
    );

    CUDA_CHECK_ERRORS();
}


void activation_increment_cuda_wrapper(
    torch::Tensor &X,
    torch::Tensor const &in_incr,
    torch::Tensor &out_incr  // expect a zero tensor
){

    CHECK_INPUT(X);
    CHECK_INPUT(in_incr);
    CHECK_INPUT(out_incr);

    activation_increment_cuda<float, 16, 3, 3>(
        X,
        in_incr,
        out_incr  // expect a zero tensor
    );
}

