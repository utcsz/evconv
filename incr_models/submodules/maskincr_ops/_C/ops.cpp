#include <c10/cuda/CUDAStream.h>
#include "ops.h"
#include "conv_wrappers.h"
#include "utils.h"
#include "checks.h"

//dispatch cuda kernel
void activation_increment(
    torch::Tensor &X,
    torch::Tensor const &in_incr,
    torch::Tensor &out_incr  // expect a zero tensor
)
{
    CHECK_INPUT(X);
    CHECK_INPUT(in_incr);
    CHECK_INPUT(out_incr);

    activation_increment_cuda_wrapper(
      X,
      in_incr,
      out_incr  // expect a zero tensor
    );
}


void conv_template(
    torch::Tensor const &x_incr,
    torch::Tensor const &mask_load,
    torch::Tensor const &mask_compute,
    torch::Tensor const &filter,
    torch::Tensor &out_incr,
    int filter_size
){
    CHECK_CUDA(x_incr);   //NOT CONTIGUOUS
    CHECK_CUDA(mask_load);   // contiguous;
    CHECK_CUDA(mask_compute);   // contiguous;
    CHECK_INPUT(filter);   // contiguous;
    CHECK_CUDA(out_incr); // not contiguous

    conv_cuda_wrapper(
      x_incr,
      mask_load,
      mask_compute,
      filter,
      out_incr,
      filter_size
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("activation_increment" , &activation_increment , "activate and increment x tensor;");
  m.def("conv_template", &conv_template, "convolution kernel;");
}


