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
    int STEP_out_C, // assertion: same as number of threads per block: required
    int OUT_CHANNELS_PER_BLOCK> // assertion: multiple of STEP_out_C: required
__global__ void conv2dkernel(
    scalar_t const * const input,
    int const * const mask_load,
    int const * const mask_compute,
    scalar_t const * const filter, // HWIO
    scalar_t * __restrict__ output,
    int const in_C, int const in_H, int const in_W,
    int const out_C, int const out_H, int const out_W
){
    int const bw = blockIdx.x;
    int const bh = blockIdx.y;
    int const bc = blockIdx.z;
    int const th = threadIdx.x; // until step_out_c
    int const warp_lane = th%WARPSIZE;
    int const warp_id = th/WARPSIZE;

    int const IN_BLOCK_W = BLOCK_W + K - 1;
    int const IN_BLOCK_H = BLOCK_H + K - 1;

    __shared__ scalar_t input_shared[IN_BLOCK_H][IN_BLOCK_W][WARPSIZE];
    __shared__ int mask_load_shared[IN_BLOCK_H][IN_BLOCK_W];

    __shared__ scalar_t output_shared[BLOCK_H][BLOCK_W][STEP_out_C];
    __shared__ int mask_compute_shared[BLOCK_H][BLOCK_W];

    for(int c_base_out = bc*OUT_CHANNELS_PER_BLOCK; c_base_out < (1+bc)*OUT_CHANNELS_PER_BLOCK && c_base_out < out_C; c_base_out += STEP_out_C){

        int out_c = c_base_out + th;

        __syncthreads();
        for(int h = 0; h < BLOCK_H; h++){
            for(int w = 0; w < BLOCK_W; w++){
                output_shared[h][w][th] = 0.0f; // computed at a bunch  of threads at a time
            }
        }


        // cooperative fetch input of warp size length
        for(int in_c_base = 0; in_c_base < in_C; in_c_base += WARPSIZE){


            // only first 32 threads active during load
            __syncthreads();

            // load mask
            for(int h = 0; h < IN_BLOCK_H; h++){
                for(int w = 0; w < IN_BLOCK_W; w++){
                    int in_c = in_c_base + warp_lane;
                    int in_h = bh*BLOCK_H + h;
                    int in_w = bw*BLOCK_W + w;
                    const bool valid = in_c < in_C && in_h < in_H && in_w < in_W;
                    mask_load_shared[h][w] = valid ? mask_load[ (in_h*in_W + in_w)*in_C + in_c] : 0;
                }
            }
            for(int h = 0; h < BLOCK_H; h++){
                for(int w = 0; w < BLOCK_W; w++){
                    int in_c = in_c_base + warp_lane;
                    int in_h = bh*BLOCK_H + h;
                    int in_w = bw*BLOCK_W + w;
                    const bool valid = in_c < in_C && in_h < in_H && in_w < in_W;
                    mask_compute_shared[h][w] = valid ? mask_compute[ (in_h*in_W + in_w)*in_C + in_c] : 0;
                }
            }


            for(int h = 0; h < IN_BLOCK_H; h++){
                for(int w = 0; w < IN_BLOCK_W; w++){
                    int in_c = in_c_base + warp_lane;
                    int in_h = bh*BLOCK_H + h;
                    int in_w = bw*BLOCK_W + w;
                    const bool valid = in_c < in_C && in_h < in_H && in_w < in_W && (mask_load_shared[h][w] != 0);
                    input_shared[h][w][warp_lane] = valid ? input[ (in_h*in_W + in_w)*in_C + in_c] : 0.0f;
                }
            }
            __syncthreads();
            // all threads active
            if(out_c < out_C){
                for(int in_c = in_c_base; in_c < in_C && in_c < in_c_base + WARPSIZE; in_c++){
                    // initialize output
                    // load filter
                    scalar_t filter_reg[K][K];
                    for(int h = 0; h < K; h++){
                        for(int w = 0; w < K; w++){
                            filter_reg[h][w] = filter[out_c*in_C*K*K +in_c*K*K + h*K + w];
                        }
                    }

                    // compute output: main kernel impl
                    #pragma unroll
                    for(int hout = 0; hout < BLOCK_H; hout++){
                        #pragma unroll
                        for(int wout = 0; wout < BLOCK_W; wout++){

                            // actual filter computation: skip if required
                            scalar_t outt = 0.0f;
                            for(int h = 0; h < K; h++){
                                for(int w = 0; w < K; w++){
                                    outt += input_shared[hout+h][wout+w][warp_lane] * filter_reg[h][w];
                                }
                            }
                            output_shared[hout][wout][th] = outt;
                        }
                    }
                }
            }
        }


        __syncthreads();

        if(out_c < out_C){
            //store computed output cooperatively
            for(int h = 0; h < BLOCK_H; h++){
                for(int w = 0; w < BLOCK_W; w++){

                    int out_h = bh*BLOCK_H + h;
                    int out_w = bw*BLOCK_W + w;
                    output[(out_h*out_W + out_w)*out_C + out_c] = output_shared[h][w][th];

                }
            }            
        }
        __syncthreads();
    } // cout loop
}


static void convconfq(
    float const * const in_incr,
    int const * const mask_load,
    int const * const mask_compute,
    float const * const filter,
    float * __restrict__ out_incr,
    int const in_C, int const in_H, int const in_W,
    int const out_C, int const out_H, int const out_W
){
    int constexpr K = 3;
    int constexpr BLOCK_W = 6;
    int constexpr BLOCK_H = 6;
    int constexpr OUT_CHANNELS_PER_BLOCK = 256;
    int constexpr STEP_out_C = 256;

    int const threads = 256;
    int const blocks_w = (out_W + BLOCK_W - 1)/BLOCK_W;
    int const blocks_h = (out_H + BLOCK_H - 1)/BLOCK_H;
    int const blocks_c = (out_C + OUT_CHANNELS_PER_BLOCK - 1)/threads;
    dim3 const blocks(blocks_w, blocks_h, blocks_c);

    conv2dkernel<
    float,
    K, // kernel size
    BLOCK_H,BLOCK_W, // block_h, block_w
    STEP_out_C, // Cout step size
    OUT_CHANNELS_PER_BLOCK> // oUt channels per block
    <<<blocks, threads>>>(
        in_incr,
        mask_load,
        mask_compute,
        filter,
        out_incr,
        in_C, in_H, in_W,
        out_C, out_H, out_W
    );
} 


void conv_cuda_wrapper(
    torch::Tensor const &in_incr,
    torch::Tensor const &mask_load,
    torch::Tensor const &mask_compute,
    torch::Tensor const &filter,
    torch::Tensor &out_incr,  // empty tensor;
    int kernel_size
){
    int const in_C = in_incr.size(1);
    int const in_H = in_incr.size(2);
    int const in_W = in_incr.size(3);

    int const out_C = out_incr.size(1);
    int const out_H = out_incr.size(2);
    int const out_W = out_incr.size(3);

    if(kernel_size == 3) {
        convconfq(
            in_incr.data_ptr<float>(),
            mask_load.data_ptr<int>(),
            mask_compute.data_ptr<int>(),
            filter.data_ptr<float>(),
            out_incr.data_ptr<float>(),
            in_C, in_H, in_W,
            out_C, out_H, out_W
        );
    }
}


