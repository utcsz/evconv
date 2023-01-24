#pragma once
#include <torch/extension.h>

// activate and increment wrapper
void activation_increment_cuda_wrapper(
    torch::Tensor &X,
    torch::Tensor const &in_incr,
    torch::Tensor &out_incr  // expect a zero tensor
);




// conv wrapper
void conv_cuda_wrapper(
    torch::Tensor const &in_incr,
    torch::Tensor const &mask_load,
    torch::Tensor const &mask_compute,
    torch::Tensor const &filter,
    torch::Tensor &out_incr,  // expect a zero tensor
    int k
);






