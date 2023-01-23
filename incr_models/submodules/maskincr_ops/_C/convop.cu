#include <torch/extension.h>
#include <conv_wrappers.h>


#define WARPSIZE 32

// valid convolution algorithm
// skip over skipslice_h and skipslice_w pixels if they are zero
// choose skipslice_h = 4, skipslice_w = 8, which is also the block configuration
template<
    typename scalar_t,
    int K,
    int BLOCK_H, int BLOCK_W,
    int SKIPSLICE_H,int SKIPSLICE_W,
    int STEP_out_C>
__global__ conv2dkernel(
    scalar_t const * const input,
    scalar_t const * const mask,
    scalar_t const * const filter,
    scalar_t * __restrict__ output,
    int const in_C, int const in_H, int const in_W,
    int const out_C, int const out_H, int const out_W,
){

    int const bw = blockIdx.x;
    int const bh = blockIdx.y;
    int const bc = blockIdx.z;
    int const th = threadIdx.x;
    int const warp_lane = th%WARPSIZE;
    int const warp_id = th/WARPSIZE;

    __shared__ scalar_t input_shared[OUT_CHANNELS_PER_BLOCK][IN_CHANNELS_PER_BLOCK][K][K];
    __shared__ scalar_t output_shared[OUT_CHANNELS_PER_BLOCK][IN_CHANNELS_PER_BLOCK][K][K];

    for(int c_base = bc*OUT_CHANNELS_PER_BLOCK; c_base < (1+bc)*OUT_CHANNELS_PER_BLOCK && c_base < out_C; c_base += STEP_out_C){

        __syncthreads();

        // cooperative fetch input of warp size length
        __shared__ scalar_t input_shared[BLOCK_H][BLOCK_W];
        for(int c = 0; c < in_C; c += WARPSIZE){
            for(int h = 0; h < BLOCK_H; h++){
                for(int w = 0; w < BLOCK_W; w+=WARPSIZE){
                    input_shared[warp_lane] = input[( (bh+h)*in_W + (bw+w))*in_C + (bc+c)];
                }
            }
        }
        __syncthreads();

        
        // initialize output
        for(int c = 0; c < in_C; c++){
            // load filter
            __shared__ filter_shared[K][K];
            for(int h = 0; h < K; h++){
                for(int w = 0; w < K; w++){
                    filter_shared[h][w] = filter[c_inner*KERNEL_SIZE*KERNEL_SIZE + thread_id];
                }
            }
        }

        //store computed output
        for(int c = 0; c < in_C; c++){
            output[c_base*in_C + c] = output_shared[c_base][c];
        }
        __syncthreads();

    }

}


void conv_cuda_wrapper(
    torch::Tensor const &in_incr,
    torch::Tensor const &mask,
    torch::Tensor const &filter,
    torch::Tensor &out_incr,  // empty tensor;
    int kernel_size
){

    conv2dkernel<
    float,
    3,
    8,4>
    <<<>>>(
            
    );
}


