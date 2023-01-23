#pragma once

#include <torch/extension.h>

#define divup(a, b) (((a) + (b) - 1) / (b))



namespace Utils{
    template <typename T>
    constexpr T constexpr_min(const T a, const T b) {
        return a > b ? b : a;
    }
    template <typename T>
    constexpr T constexpr_max(const T a, const T b) {
        return a < b ? b : a;
    }
};


// trivially copyable structs
struct dim{
    int const C, H, W;
    dim(c10::IntArrayRef sizes): C(sizes[1]), H(sizes[2]), W(sizes[3])
    {
    }
};


