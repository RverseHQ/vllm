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
