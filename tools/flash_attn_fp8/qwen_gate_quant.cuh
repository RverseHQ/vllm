// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cutlass/numeric_types.h>

namespace flash {

// A full warp owns one group of 128 final-output values, four per lane.
// The caller has already rounded the attention result to BF16.
__device__ __forceinline__ void qwen_gate_quant_warp(
    float const (&attention)[4], cutlass::bfloat16_t const* gate,
    cutlass::float_e4m3_t* output, float* scale_output) {
  float product[4];
  float absmax = 1e-10f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float sigmoid = __fdividef(1.f, 1.f + __expf(-float(gate[i])));
    // Match the compiled Qwen gate: sigmoid stays FP32; only the
    // product is rounded to BF16 before the existing FP8 quantization.
    product[i] = float(cutlass::bfloat16_t(attention[i] * sigmoid));
    absmax = fmaxf(absmax, fabsf(product[i]));
  }
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    absmax = fmaxf(absmax, __shfl_xor_sync(0xffffffff, absmax, delta));
  }
  // Match per_token_group_fp8_quant's CUDA E8M0 path, including its
  // post-division epsilon clamp (not the Triton fallback's pre-clamp).
  float raw_scale = fmaxf(absmax / 448.f, 1e-10f);
  unsigned bits = __float_as_uint(raw_scale);
  unsigned exponent = (bits >> 23) & 0xffu;
  float scale = __uint_as_float((exponent + ((bits & 0x7fffffu) != 0u)) << 23);
  if ((threadIdx.x & 31) == 0) {
    *scale_output = scale;
  }
  float inv_scale = 1.f / scale;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    output[i] = cutlass::float_e4m3_t(
        fminf(fmaxf(product[i] * inv_scale, -448.f), 448.f));
  }
}

}  // namespace flash
