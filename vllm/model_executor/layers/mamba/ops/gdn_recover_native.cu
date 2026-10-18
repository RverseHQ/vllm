/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 * Derived from vLLM csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu.
 * FP32 state specialization; state arithmetic and reduction order retained.
 */
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <initializer_list>

namespace {
constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int H = 16;
constexpr int HV = 48;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kChunkV = 32;
constexpr int kNumChunks = kDimV / kChunkV;
constexpr int kRowsPerWarp = kChunkV / kWarps;
constexpr int kMaxMtpTokens = 8;

struct GdnRecoverStrides {
  int64_t mixed_row;
  int64_t a_row;
  int64_t b_row;
  int64_t state_slot;
};

// Fixed int64 ABI shared with GdnRecoverLayerDescriptor in
// gdn_recover.py.
struct GdnRecoverLayerDescriptor {
  int64_t mixed_qkv, a, b, a_log, dt_bias, state, conv;
  int64_t mixed_row, a_row, b_row, state_slot, dt_bias_type;
  int64_t conv_page, conv_channel, conv_time, history;
};
static_assert(sizeof(GdnRecoverLayerDescriptor) == 16 * sizeof(int64_t));

__device__ __forceinline__ void copy_state_chunk(float* shared_state,
                                                 const float* state, int chunk,
                                                 int thread) {
  constexpr int kElementsPerCopy = 4;
  constexpr int kCopiesPerChunk = kChunkV * kDimK / kElementsPerCopy;
  const int stage = chunk % 2;
  for (int copy = thread; copy < kCopiesPerChunk; copy += kThreads) {
    const int element = copy * kElementsPerCopy;
    const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(
        shared_state + stage * kChunkV * kDimK + element));
    const float* gmem = state + chunk * kChunkV * kDimK + element;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :
                 : "r"(smem_addr), "l"(gmem));
  }
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ float sigmoid_fast(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float softplus_fast(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}

__device__ __forceinline__ float load_dt_bias(const void* dt_bias, int head,
                                              int dt_bias_type) {
  if (dt_bias_type == 1) {
    return __bfloat162float(static_cast<const __nv_bfloat16*>(dt_bias)[head]);
  }
  if (dt_bias_type == 2) {
    return __half2float(static_cast<const __half*>(dt_bias)[head]);
  }
  return static_cast<const float*>(dt_bias)[head];
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

struct Sum2 {
  float x;
  float y;
};

__device__ __forceinline__ Sum2 warp_reduce_sum_pair(float x, float y) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffffu, x, offset);
    y += __shfl_xor_sync(0xffffffffu, y, offset);
  }
  return {x, y};
}

template <bool Batched>
__global__ __launch_bounds__(kThreads, 2) void gdn_recover_native_kernel(
    const __nv_bfloat16* __restrict__ mixed_qkv,
    const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ a_log, const void* __restrict__ dt_bias,
    const int* __restrict__ state_indices, const int* __restrict__ cu_seqlens,
    const int* __restrict__ source_selectors, float* __restrict__ state,
    int state_indices_width, int dt_bias_type, GdnRecoverStrides strides,
    const GdnRecoverLayerDescriptor* __restrict__ descriptors) {
  if constexpr (Batched) {
    const GdnRecoverLayerDescriptor d = descriptors[blockIdx.z];
    mixed_qkv = reinterpret_cast<const __nv_bfloat16*>(d.mixed_qkv);
    a = reinterpret_cast<const __nv_bfloat16*>(d.a);
    b = reinterpret_cast<const __nv_bfloat16*>(d.b);
    a_log = reinterpret_cast<const float*>(d.a_log);
    dt_bias = reinterpret_cast<const void*>(d.dt_bias);
    state = reinterpret_cast<float*>(d.state);
    dt_bias_type = d.dt_bias_type;
    strides = {d.mixed_row, d.a_row, d.b_row, d.state_slot};
  }
  const int request = blockIdx.x;
  const int value_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bos = cu_seqlens[request];
  const int eos = cu_seqlens[request + 1];
  int num_tokens = eos - bos;
  if (num_tokens <= 0 || num_tokens > kMaxMtpTokens) return;
  const int selector = source_selectors[request];
  const int source_slot =
      selector > 0 && selector <= state_indices_width
          ? state_indices[request * state_indices_width + selector - 1]
          : 0;
  if (source_slot <= 0) return;
  while (num_tokens > 0 &&
         (num_tokens == selector ||
          state_indices[request * state_indices_width + num_tokens - 1] <= 0)) {
    --num_tokens;
  }
  if (num_tokens == 0) return;

  const int key_head = value_head / (HV / H);
  __shared__ float shared_state[2][kChunkV][kDimK];
  __shared__ float shared_k[kMaxMtpTokens][kDimK];
  __shared__ __nv_bfloat16 shared_v[kMaxMtpTokens][kDimV];
  __shared__ float shared_decay[kMaxMtpTokens];
  __shared__ float shared_beta[kMaxMtpTokens];
  float* source_state = state +
                        static_cast<int64_t>(source_slot) * strides.state_slot +
                        value_head * kDimV * kDimK;
  copy_state_chunk(&shared_state[0][0][0], source_state, 0, tid);

  if (warp < num_tokens) {
    const int t = warp;
    const int token = bos + t;
    const int64_t mixed_base = static_cast<int64_t>(token) * strides.mixed_row;
    float k_values[4];
    float k_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      k_values[i] = __bfloat162float(
          mixed_qkv[mixed_base + H * kDimK + key_head * kDimK + dim]);
      shared_v[t][dim] =
          mixed_qkv[mixed_base + 2 * H * kDimK + value_head * kDimV + dim];
      k_square += k_values[i] * k_values[i];
    }
    const float k_sum = warp_reduce_sum(k_square);
    const float k_scale =
        __shfl_sync(0xffffffffu, lane == 0 ? rsqrtf(k_sum + 1.0e-6f) : 0.0f, 0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      shared_k[t][dim] = k_values[i] * k_scale;
    }
    if (lane == 0) {
      const float a_value = __bfloat162float(
          a[static_cast<int64_t>(token) * strides.a_row + value_head]);
      const float b_value = __bfloat162float(
          b[static_cast<int64_t>(token) * strides.b_row + value_head]);
      const float g = -__expf(a_log[value_head]) *
                      softplus_fast(a_value + load_dt_bias(dt_bias, value_head,
                                                           dt_bias_type));
      shared_decay[t] = __expf(g);
      shared_beta[t] = sigmoid_fast(b_value);
    }
  }
  __syncthreads();
  const int k_base = lane * 4;
  int rows[kRowsPerWarp];
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) {
    rows[row] = warp + row * kWarps;
  }
#pragma unroll
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();
    if (chunk + 1 < kNumChunks) {
      copy_state_chunk(&shared_state[0][0][0], source_state, chunk + 1, tid);
    }
    float h[kRowsPerWarp][4];
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
      const float4 state_value = *reinterpret_cast<const float4*>(
          &shared_state[chunk & 1][rows[row]][k_base]);
      h[row][0] = state_value.x;
      h[row][1] = state_value.y;
      h[row][2] = state_value.z;
      h[row][3] = state_value.w;
    }
    for (int t = 0; t < num_tokens; ++t) {
      const float4 k4 = *reinterpret_cast<const float4*>(&shared_k[t][k_base]);
      const float k_values[4] = {k4.x, k4.y, k4.z, k4.w};
      float dot_hk[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] *= shared_decay[t];
          dot_hk[row] += h[row][i] * k_values[i];
        }
      }
      const Sum2 dot_hk_01 = warp_reduce_sum_pair(dot_hk[0], dot_hk[1]);
      const Sum2 dot_hk_23 = warp_reduce_sum_pair(dot_hk[2], dot_hk[3]);
      const float reduced_hk[kRowsPerWarp] = {dot_hk_01.x, dot_hk_01.y,
                                              dot_hk_23.x, dot_hk_23.y};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        const int value = chunk * kChunkV + rows[row];
        const float delta =
            (__bfloat162float(shared_v[t][value]) - reduced_hk[row]) *
            shared_beta[t];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] += k_values[i] * delta;
        }
      }
      const int destination_slot =
          state_indices[request * state_indices_width + t];
      if (destination_slot > 0) {
        float* destination_state =
            state +
            static_cast<int64_t>(destination_slot) * strides.state_slot +
            value_head * kDimV * kDimK;
#pragma unroll
        for (int row = 0; row < kRowsPerWarp; ++row) {
          const int value = chunk * kChunkV + rows[row];
          *reinterpret_cast<float4*>(destination_state + value * kDimK +
                                     k_base) =
              make_float4(h[row][0], h[row][1], h[row][2], h[row][3]);
        }
      }
    }
  }
}

void recover_native(const at::Tensor& mixed_qkv, const at::Tensor& a,
                    const at::Tensor& b, const at::Tensor& a_log,
                    const at::Tensor& dt_bias, const at::Tensor& state_indices,
                    const at::Tensor& cu_seqlens,
                    const at::Tensor& source_selectors, at::Tensor state) {
  TORCH_CHECK(mixed_qkv.is_cuda() && mixed_qkv.scalar_type() == at::kBFloat16 &&
                  mixed_qkv.dim() == 2 && mixed_qkv.stride(1) == 1,
              "mixed_qkv must be CUDA BF16 with contiguous channels");
  TORCH_CHECK(state.is_cuda() && state.scalar_type() == at::kFloat &&
                  state.dim() == 4 && state.size(2) == kDimV &&
                  state.size(3) == kDimK && state.stride(1) == kDimV * kDimK &&
                  state.stride(2) == kDimK && state.stride(3) == 1 &&
                  state.stride(0) >= state.size(1) * kDimV * kDimK &&
                  state.stride(0) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0,
              "state-only native recovery requires aligned FP32 state slots");
  TORCH_CHECK(state.size(1) == HV && mixed_qkv.size(1) == (2 * H + HV) * kDimK,
              "native recovery candidate requires TP1 Hk=16/Hv=48");
  for (const auto* tensor : {&a, &b}) {
    TORCH_CHECK(tensor->device() == mixed_qkv.device() &&
                    tensor->scalar_type() == at::kBFloat16 &&
                    tensor->dim() == 2 &&
                    tensor->size(0) == mixed_qkv.size(0) &&
                    tensor->size(1) == HV && tensor->stride(1) == 1,
                "a/b must be BF16 [tokens, HV]");
  }
  TORCH_CHECK(a_log.device() == mixed_qkv.device() &&
                  a_log.scalar_type() == at::kFloat && a_log.is_contiguous() &&
                  a_log.numel() == HV,
              "A_log must be contiguous FP32 [HV]");
  TORCH_CHECK(dt_bias.device() == mixed_qkv.device() &&
                  dt_bias.is_contiguous() && dt_bias.numel() == HV &&
                  (dt_bias.scalar_type() == at::kFloat ||
                   dt_bias.scalar_type() == at::kBFloat16 ||
                   dt_bias.scalar_type() == at::kHalf),
              "dt_bias must be contiguous FP32/BF16/FP16 [HV]");
  for (const auto* tensor : {&state_indices, &cu_seqlens, &source_selectors}) {
    TORCH_CHECK(tensor->device() == mixed_qkv.device() &&
                    tensor->scalar_type() == at::kInt &&
                    tensor->is_contiguous(),
                "recovery metadata must be contiguous CUDA int32");
  }
  TORCH_CHECK(state.device() == mixed_qkv.device(), "state device mismatch");
  TORCH_CHECK(state_indices.dim() == 2 && state_indices.size(1) > 1 &&
                  state_indices.size(1) <= kMaxMtpTokens &&
                  cu_seqlens.dim() == 1 &&
                  cu_seqlens.numel() == state_indices.size(0) + 1 &&
                  source_selectors.dim() == 1 &&
                  source_selectors.numel() == state_indices.size(0),
              "invalid replay index/selector shapes");
  if (state_indices.size(0) == 0 || mixed_qkv.size(0) == 0) return;
  const int dt_bias_type =
      dt_bias.scalar_type() == at::kFloat
          ? 0
          : (dt_bias.scalar_type() == at::kBFloat16 ? 1 : 2);
  const GdnRecoverStrides strides{mixed_qkv.stride(0), a.stride(0), b.stride(0),
                                  state.stride(0)};
  const c10::cuda::CUDAGuard device_guard(mixed_qkv.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(mixed_qkv.get_device());
  gdn_recover_native_kernel<false>
      <<<dim3(state_indices.size(0), HV), kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(mixed_qkv.data_ptr()),
          static_cast<const __nv_bfloat16*>(a.data_ptr()),
          static_cast<const __nv_bfloat16*>(b.data_ptr()),
          a_log.data_ptr<float>(), dt_bias.data_ptr(),
          state_indices.data_ptr<int>(), cu_seqlens.data_ptr<int>(),
          source_selectors.data_ptr<int>(), state.data_ptr<float>(),
          state_indices.size(1), dt_bias_type, strides, nullptr);
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess,
              "GDN state-only launch failed: ", cudaGetErrorString(error));
}

__global__ void gdn_compact_conv_batch_kernel(
    const GdnRecoverLayerDescriptor* __restrict__ descriptors,
    const int* __restrict__ plan) {
  const GdnRecoverLayerDescriptor d = descriptors[blockIdx.z];
  const int* row = plan + blockIdx.x * 5;
  const int accepted = row[4];
  if (accepted <= 0) return;
  const int64_t source = row[0], prefix = row[1], final = row[3];
  const int prefix_count = row[2];
  const int channel = blockIdx.y * kThreads + threadIdx.x;
  auto* state = reinterpret_cast<__nv_bfloat16*>(d.conv);
  __nv_bfloat16 final_window[3], prefix_window[3];
#pragma unroll
  for (int t = 0; t < 3; ++t) {
    const int64_t offset = channel * d.conv_channel + t * d.conv_time;
    final_window[t] =
        state[source * d.conv_page + offset + (accepted - 1) * d.conv_time];
    if (prefix_count > 0) {
      prefix_window[t] = state[source * d.conv_page + offset +
                               (prefix_count - 1) * d.conv_time];
    }
  }
  // All reads precede writes, including when prefix/final aliases the source.
  __syncthreads();
#pragma unroll
  for (int t = 0; t < 3; ++t) {
    const int64_t offset = channel * d.conv_channel + t * d.conv_time;
    if (prefix_count > 0) {
      state[prefix * d.conv_page + offset] = prefix_window[t];
    }
    state[final * d.conv_page + offset] = final_window[t];
  }
}

void check_descriptors(const at::Tensor& descriptors) {
  TORCH_CHECK(descriptors.is_cuda() && descriptors.scalar_type() == at::kLong &&
                  descriptors.dim() == 2 && descriptors.size(0) > 0 &&
                  descriptors.size(1) == 16 && descriptors.is_contiguous(),
              "batch descriptors must be contiguous CUDA int64 [layers, 16]");
}

void recover_native_batch(at::Tensor descriptors,
                          const at::Tensor& state_indices,
                          const at::Tensor& cu_seqlens,
                          const at::Tensor& source_selectors) {
  check_descriptors(descriptors);
  for (const auto* tensor : {&state_indices, &cu_seqlens, &source_selectors}) {
    TORCH_CHECK(tensor->device() == descriptors.device() &&
                    tensor->scalar_type() == at::kInt &&
                    tensor->is_contiguous(),
                "recovery metadata must be contiguous CUDA int32");
  }
  TORCH_CHECK(state_indices.dim() == 2 && state_indices.size(1) == 5 &&
                  cu_seqlens.dim() == 1 &&
                  cu_seqlens.numel() == state_indices.size(0) + 1 &&
                  source_selectors.dim() == 1 &&
                  source_selectors.numel() == state_indices.size(0),
              "native batch recovery requires MTP3 replay metadata");
  if (state_indices.size(0) == 0) return;
  const c10::cuda::CUDAGuard device_guard(descriptors.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(descriptors.get_device());
  gdn_recover_native_kernel<true>
      <<<dim3(state_indices.size(0), HV, descriptors.size(0)), kThreads, 0,
         stream>>>(nullptr, nullptr, nullptr, nullptr, nullptr,
                   state_indices.data_ptr<int>(), cu_seqlens.data_ptr<int>(),
                   source_selectors.data_ptr<int>(), nullptr,
                   state_indices.size(1), 0, {},
                   reinterpret_cast<const GdnRecoverLayerDescriptor*>(
                       descriptors.data_ptr<int64_t>()));
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess,
              "GDN batch recovery failed: ", cudaGetErrorString(error));
}

void compact_conv_batch(at::Tensor descriptors, const at::Tensor& plan) {
  check_descriptors(descriptors);
  TORCH_CHECK(plan.device() == descriptors.device() &&
                  plan.scalar_type() == at::kInt && plan.dim() == 2 &&
                  plan.size(1) == 5 && plan.is_contiguous(),
              "convolution plan must be contiguous CUDA int32 [requests, 5]");
  if (plan.size(0) == 0) return;
  const c10::cuda::CUDAGuard device_guard(descriptors.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(descriptors.get_device());
  constexpr int kChannels = (2 * H + HV) * kDimK;
  static_assert(kChannels % kThreads == 0);
  gdn_compact_conv_batch_kernel<<<dim3(plan.size(0), kChannels / kThreads,
                                       descriptors.size(0)),
                                  kThreads, 0, stream>>>(
      reinterpret_cast<const GdnRecoverLayerDescriptor*>(
          descriptors.data_ptr<int64_t>()),
      plan.data_ptr<int>());
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess,
              "GDN batch convolution failed: ", cudaGetErrorString(error));
}
}  // namespace

TORCH_LIBRARY(gdn_recover_state_only, m) {
  m.def(
      "native(Tensor mixed_qkv, Tensor a, Tensor b, Tensor A_log, Tensor "
      "dt_bias, "
      "Tensor state_indices, Tensor cu_seqlens, Tensor source_selectors, "
      "Tensor(a!) state) -> ()");
  m.def(
      "native_batch(Tensor(a!) descriptors, Tensor state_indices, "
      "Tensor cu_seqlens, Tensor source_selectors) -> ()");
  m.def("conv_batch(Tensor(a!) descriptors, Tensor plan) -> ()");
}
TORCH_LIBRARY_IMPL(gdn_recover_state_only, CUDA, m) {
  m.impl("native", &recover_native);
  m.impl("native_batch", &recover_native_batch);
  m.impl("conv_batch", &compact_conv_batch);
}
