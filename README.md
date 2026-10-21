# vLLM：Qwen3.8-27B-FP8 / H800 优化

基于 [vLLM `237d99b4`](https://github.com/vllm-project/vllm/commit/237d99b4aff67ecfedacca7acb0ee93f299fa54e)。优先减少非 KV 显存占用、扩大 KV 池，其次加快推理；不新增量化、不缩短上下文、不减少 MTP，也不裁掉视觉或工具能力。

## 实测效果

同一张 H800 PCIe 80GB、同一模型和冻结负载下，原生主线复测与保留优化方案的对比：

| 指标 | 原生主线 | 优化后 | 变化 |
| --- | ---: | ---: | ---: |
| KV 池容量（GiB） | 41.35 | 45.60 | +4.25 GiB / +10.29% |
| KV 总容量（tokens） | 1,147,960 | 1,331,933 | +16.03% |
| 整批输出吞吐（tokens/s） | 203.38 | 294.36 | +44.73% |
| 整批耗时（s） | 327.27 | 226.12 | -30.91% |
| NVML 采样显存峰值（MiB） | 80,400.81 | 78,210.81 | -2,190 MiB |

测试条件：

- 模型 `Qwen3.8-27B-FP8`，revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`。
- 单卡 TP1 / DP1 / PP1，显存预算 0.94；最大上下文 262,144，`max_num_seqs=48`，`max_num_batched_tokens=16384`，MTP3。
- 保持 FP8 权重和 KV、BF16 卷积状态、FP32 SSM 状态；启用前缀缓存，保留完整视觉和工具配置。
- PyTorch `2.14.0+cu130`、Triton `3.8.0`、FlashInfer `0.7.2`。对照使用 `237d99b4` Python 源码，基础原生内核来自编译输入一致的 `f98ae6a9` 预编译 wheel。
- 冻结的 10 请求长上下文混合负载：8 个解码请求与 2 个 prefill 请求，共输出 66,560 tokens；两组均全部完成，抢占次数均为 0。优化组加载已有 AOT 编译产物，并启用独立 FA3 库。

KV 总容量是按 262K 上下文口径估计的混合 KV/SSM 池容量，不是单请求最大长度；整批吞吐不是单流解码速度。结果来自保留方案的 GPU 实验，fork 集成和目录整理后未重新跑 GPU。总收益是整套优化的观测，不按单项分摊。

## 优化措施

| 措施 | 实现与作用 |
| --- | --- |
| GDN 紧凑恢复 | 用原始输入 replay 替代逐草稿状态快照，按 3 组 × 16 层批量恢复；保留 FP32 SSM、BF16 卷积状态、prefill checkpoint 和前缀缓存状态语义。 |
| 原值 UVA 卸载 | 将输入 embedding 和视觉权重放到 pinned 主存，通过 UVA 访问；输出 `lm_head` 留在 GPU。释放权重显存，代价是主存占用和 PCIe 访问。 |
| MLP 临时内存 | 按 2048 tokens 均衡分块，融合 SiLU 乘法与既有 FP8 group quantization，并将 down projection 直接写入最终输出。 |
| QKV 临时内存 | 按 4096 tokens 分块，融合 QK normalization、RoPE 和 gate 拆分，直接写入最终 Q/K/V/gate 缓冲区。 |
| 视觉临时内存 | 视觉 MLP 分块、packed rotary、独占 GELU/residual 缓冲区复用，以及最后使用后释放中间张量。 |
| 编译与显存预算 | 在 cached AOT 的 piecewise 执行路径释放最后使用的输入；完整测量 FULL CUDA Graph 显存而非抽样外推；在适用的单 worker 配置启用 expandable segments。 |
| 长上下文 FA3 | 保持原 FP8 格式、split 策略和两级 FP32 PV 累加，使用真正的 MTP Q4 M64/N128 内核；其他形状及 BF16 保留 SDK 路径。独立库仍需单独构建并显式加载。 |

## 输出融合与变长验证实验

以下路径默认关闭，不计入上表收益；H800 整模型数值、吞吐和 KV 容量均为 **Not tested**。

| 开关 | 作用与范围 |
| --- | --- |
| `VLLM_QWEN_GDN_FP8_OUTPUT=1` | 将原生 GDN 递推、门控归一化的最终输出直接量化，供既有 DeepGEMM 投影使用；仅 TP1、纯 speculative decode、FP32 state、group128/E8M0。 |
| `VLLM_QWEN_FA3_FP8_OUTPUT=1` | 在既有 Q4 split final-combine 中融合输出门控与量化；仅本仓 page1600 布局、`VLLM_COMPILE`/Inductor 编译路径。eager 保留原 sigmoid 舍入语义。 |
| `VLLM_QWEN_GDN_DEVICE_METADATA=1` | 将纯解码的状态槽、设备查询边界和 padding 元数据合并为一次设备端准备；可独立用于固定 MTP3 对照。 |

需重新编译本 fork 的原生扩展。小 batch 的 FlashInfer BF16 输入路由不变；LoRA 等声明需要原始激活的投影包装层、mixed/prefill 和不适用配置保留原路径。

FA3 实验还需构建独立库。按源码 manifest 固定 `FA3_SOURCE_DIR`、`CUTLASS_SOURCE_DIR`，选择持久的 `FA3_GATE_BUILD_DIR`；原有 `QWEN_MAIN237_FA3_LIBRARY` 和 `QWEN_MAIN237_FA3_Q4_LIBRARY` 保持原设置：

```bash
python tools/flash_attn_fp8/build_fp8_q4.py \
  --source-dir "$FA3_SOURCE_DIR" --cutlass-dir "$CUTLASS_SOURCE_DIR" \
  --gate-quant --output-dir "$FA3_GATE_BUILD_DIR"
export QWEN_MAIN237_FA3_Q4_GATE_QUANT_LIBRARY="$FA3_GATE_BUILD_DIR/build/_qwen_q4_gate_quant_C.so"
```

缺少对应原生算子/库、split 为 0/1 或形状不适用时，不会命中原生输出融合。不能把实验开关开启等同于融合已执行。

自适应验证复用现有在线接受率估计器，无需新置信度头；在原 MTP 配置中加入：

```json
{
  "method": "mtp",
  "num_speculative_tokens": 3,
  "rejection_sample_method": "standard",
  "enable_adaptive_verification": true,
  "use_local_argmax_reduction": false
}
```

这会自动启用 GDN 设备元数据路径，最大验证窗口仍为 4（主 token 加 3 个草稿）。混合 prefill 保留原处理，但不再信任 CPU 均分产生的逐请求 uniform 长度提示。

本机 SM89 已执行生产 GDN CUDA 源码与旧 SDK 的 17 组逐字节对照（含变长及 padding）、FA3 生产门控量化 helper、量化回退和元数据图重放；两条生产 CUDA 路径已通过 SM90 离线构建。完整 attention 与 DeepGEMM 消费者数值仍需 H800 验收。

开启现有 `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` 并导出 Torch profiler trace 后，可查看 `mrv2:prepare_attn` 和 `mrv2:forward:FULL`/`PIECEWISE`/`NONE`/`UBATCH`，区分主模型图外准备与实际图分派；未开启 profiling scopes 时这些标记为空上下文。
