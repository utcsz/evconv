#pragma once
#include <torch/extension.h>




//  incr activation kernels
void activation_increment(
    torch::Tensor &X,
    torch::Tensor const &in_incr,
    torch::Tensor &out_incr  // expect a zero tensor
);


void conv_template(
    torch::Tensor const &x_incr,
    torch::Tensor const &mask,
    torch::Tensor const &filter,
    torch::Tensor &out_incr,
    int filter_size
);




