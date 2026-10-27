"""预置结构与标称规格；仿真用有效性能及驻留权重由使用者填写。"""

from simcore.schemas import AttentionGroup, IHCSpec, ModelSpec, RouterSpec

# Qwen 官方模型卡及 config.json：
# https://huggingface.co/Qwen/Qwen3-8B
# https://huggingface.co/Qwen/Qwen3-235B-A22B
MODELS = (
    ModelSpec(
        name="Qwen/Qwen3-8B",
        num_layers=36,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        head_dim=128,
        ffn_intermediate_size=12288,
        vocab_size=151_936,
        tie_word_embeddings=False,
        attention_layout="grouped",
        attention_groups=[
            AttentionGroup(
                kind="gqa",
                layers=36,
                num_attention_heads=32,
                num_kv_heads=8,
                head_dim=128,
                qk_norm=True,
                output_gate=False,
            )
        ],
    ),
    ModelSpec(
        name="Qwen/Qwen3-235B-A22B",
        num_layers=94,
        hidden_size=4096,
        num_attention_heads=64,
        num_kv_heads=4,
        head_dim=128,
        ffn_intermediate_size=12288,
        expert_intermediate_size=1536,
        vocab_size=151_936,
        tie_word_embeddings=False,
        num_experts=128,
        experts_per_token=8,
        moe_layers=94,
        attention_layout="grouped",
        attention_groups=[
            AttentionGroup(
                kind="gqa",
                layers=94,
                num_attention_heads=64,
                num_kv_heads=4,
                head_dim=128,
                qk_norm=True,
                output_gate=False,
            )
        ],
    ),
    # BF16 主干；不计独立 MTP 草稿层。规格来自该 revision 的 config.json。
    ModelSpec(
        name="tencent/Hy4-preview",
        revision="705d81ee51566a186d645b74c974d642ef2828fe",
        num_layers=78,
        hidden_size=6144,
        num_attention_heads=64,
        num_kv_heads=8,
        head_dim=256,
        ffn_intermediate_size=18432,
        expert_intermediate_size=2048,
        vocab_size=120_832,
        tie_word_embeddings=False,
        num_experts=256,
        experts_per_token=8,
        moe_layers=77,
        num_shared_experts=1,
        shared_expert_intermediate_size=2048,
        router=RouterSpec(dtype="fp32", correction_bias=True),
        ihc=IHCSpec(mult=4),
        attention_layout="grouped",
        attention_groups=[
            AttentionGroup(
                kind="mla_dsa",
                layers=78,
                num_attention_heads=64,
                q_rank=2048,
                kv_rank=512,
                rope_dim=64,
                nope_dim=192,
                value_dim=256,
                index_heads=32,
                index_dim=128,
                topk=2048,
                full_indexer_layers=21,
                output_gate=True,
                index_norm_dtype="fp32",
                sink_dtype="fp32",
            )
        ],
    ),
    # https://huggingface.co/zai-org/GLM-5/blob/c183ef8/config.json
    # BF16 主干（含索引投影），路由修正向量为 FP32；不计独立 MTP 层。
    ModelSpec(
        name="zai-org/GLM-5",
        revision="c183ef8",
        num_layers=78,
        hidden_size=6144,
        num_attention_heads=64,
        num_kv_heads=64,
        head_dim=64,
        ffn_intermediate_size=12288,
        expert_intermediate_size=2048,
        vocab_size=154_880,
        tie_word_embeddings=False,
        num_experts=256,
        experts_per_token=8,
        moe_layers=75,
        num_shared_experts=1,
        shared_expert_intermediate_size=2048,
        router=RouterSpec(correction_bias=True),
        attention_layout="grouped",
        attention_groups=[
            AttentionGroup(
                kind="mla_dsa",
                layers=78,
                num_attention_heads=64,
                q_rank=2048,
                kv_rank=512,
                rope_dim=64,
                nope_dim=192,
                value_dim=256,
                index_heads=32,
                index_dim=128,
                topk=2048,
                full_indexer_layers=78,
            )
        ],
    ),
    # https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct/blob/main/config.json
    ModelSpec(
        name="Qwen/Qwen3-Coder-480B-A35B-Instruct",
        num_layers=62,
        hidden_size=6144,
        num_attention_heads=96,
        num_kv_heads=8,
        head_dim=128,
        ffn_intermediate_size=8192,
        expert_intermediate_size=2560,
        vocab_size=151_936,
        tie_word_embeddings=False,
        num_experts=160,
        experts_per_token=8,
        moe_layers=62,
        attention_layout="grouped",
        attention_groups=[
            AttentionGroup(
                kind="gqa",
                layers=62,
                num_attention_heads=96,
                num_kv_heads=8,
                head_dim=128,
                qk_norm=True,
            )
        ],
    ),
    # https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct
    # 结构核对：https://huggingface.co/second-state/Meta-Llama-3.1-70B-Instruct-GGUF/blob/main/config.json
    ModelSpec(
        name="meta-llama/Llama-3.1-70B-Instruct",
        num_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        num_kv_heads=8,
        head_dim=128,
        ffn_intermediate_size=28672,
        vocab_size=128_256,
        tie_word_embeddings=False,
        attention_layout="grouped",
        attention_groups=[
            AttentionGroup(
                kind="gqa",
                layers=80,
                num_attention_heads=64,
                num_kv_heads=8,
                head_dim=128,
                qk_norm=False,
            )
        ],
    ),
    # https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct
    # 结构核对（仅结构，不沿用量化）：
    # https://huggingface.co/neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w4a16/blob/main/config.json
    ModelSpec(
        name="meta-llama/Llama-3.1-405B-Instruct",
        num_layers=126,
        hidden_size=16384,
        num_attention_heads=128,
        num_kv_heads=8,
        head_dim=128,
        ffn_intermediate_size=53248,
        vocab_size=128_256,
        tie_word_embeddings=False,
        attention_layout="grouped",
        attention_groups=[
            AttentionGroup(
                kind="gqa",
                layers=126,
                num_attention_heads=128,
                num_kv_heads=8,
                head_dim=128,
                qk_norm=False,
            )
        ],
    ),
)

# 单卡显存：https://docs.nvidia.com/dgx/dgxb300-user-guide/introduction-to-dgxb300.html
# 稠密 BF16 和 HBM 标称峰值：https://inferencex.semianalysis.com/zh/chips/b300
# 单 GPU NVLink 互联标称 1.8 TB/s（双向合计）：https://www.nvidia.com/en-us/data-center/hgx/
# FP8 与 FP16/BF16 峰值算力统一按 2:1 补齐。
HARDWARE_PRESET_FIELDS = (
    "name", "memory_bytes_per_gpu", "effective_fp16_tflops", "effective_fp8_tflops",
    "effective_hbm_gb_s", "effective_interconnect_gb_s", "collective_latency_ms",
    "ep_communication_latency_ms", "kernel_overhead_ms",
)
HARDWARE_TRIAL_DEFAULTS = {
    "effective_hbm_gb_s": 1000,
    "effective_interconnect_gb_s": 600,
    "collective_latency_ms": 0.01,
    "ep_communication_latency_ms": 0.02,
    "kernel_overhead_ms": 0.005,
    "performance_note": (
        "未经实测校准：算力沿用普通配置的峰值初始化规则，仅作理想试算；"
        "HBM 1000 GB/s、单 rank 互联 600 GB/s、TP 0.01 ms/次、EP 0.02 ms/次、"
        "单算子 0.005 ms 均沿用项目通用试算假设，不代表该硬件实测性能，可单独扫描覆盖。"
    ),
}
HARDWARE = (
    {
        **HARDWARE_TRIAL_DEFAULTS,
        "name": "NVIDIA B300",
        "effective_fp16_tflops": 2250,
        "effective_fp8_tflops": 4500,
        "gpu_count": 8,
        "memory_bytes_per_gpu": 288_000_000_000,
        "peak_fp16_tflops": 2250,
        "peak_fp8_tflops": 4500,
        "peak_hbm_gb_s": 8000,
        "peak_interconnect_gb_s": 1800,
    },
    # H20 96 GB HBM3：http://www.lantek-hk.com/NVIDIA.html；4.0 TB/s 按十进制换算。
    # 稠密算力及 NVLink 双向峰值沿用 refs/KVCacheModeling/src/core/presets.js。
    {
        **HARDWARE_TRIAL_DEFAULTS,
        "name": "NVIDIA H20",
        "effective_fp16_tflops": 148,
        "effective_fp8_tflops": 296,
        "gpu_count": 8,
        "memory_bytes_per_gpu": 96_000_000_000,
        "peak_fp16_tflops": 148,
        "peak_fp8_tflops": 296,
        "peak_hbm_gb_s": 4000,
        "peak_interconnect_gb_s": 900,
    },
    # 昇腾规格沿用 refs/KVCacheModeling/src/core/presets.js。
    # 路线图：https://www.huawei.com/cn/news/2025/9/hc-xu-keynote-speech
    # FP16/BF16 按 FP8 1000 TFLOPS 的一半估算，500 TFLOPS 非官方披露值。
    {
        **HARDWARE_TRIAL_DEFAULTS,
        "name": "Huawei Ascend 950PR",
        "effective_fp16_tflops": 500,
        "effective_fp8_tflops": 1000,
        "gpu_count": 8,
        "memory_bytes_per_gpu": 128_000_000_000,
        "peak_fp16_tflops": 500,
        "peak_fp8_tflops": 1000,
        "peak_hbm_gb_s": 1600,
        "peak_interconnect_gb_s": 2000,
    },
    {
        **HARDWARE_TRIAL_DEFAULTS,
        "name": "Huawei Ascend 950DT",
        "effective_fp16_tflops": 500,
        "effective_fp8_tflops": 1000,
        "gpu_count": 8,
        "memory_bytes_per_gpu": 144_000_000_000,
        "peak_fp16_tflops": 500,
        "peak_fp8_tflops": 1000,
        "peak_hbm_gb_s": 4000,
        "peak_interconnect_gb_s": 2000,
    },
)
