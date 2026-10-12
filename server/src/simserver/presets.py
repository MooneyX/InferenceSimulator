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
        attention_groups=[AttentionGroup(
            kind="mla_dsa", layers=78, num_attention_heads=64,
            q_rank=2048, kv_rank=512, rope_dim=64, nope_dim=192, value_dim=256,
            index_heads=32, index_dim=128, topk=2048, full_indexer_layers=21,
            output_gate=True, index_norm_dtype="fp32", sink_dtype="fp32",
        )],
    ),
)

# 单卡显存：https://docs.nvidia.com/dgx/dgxb300-user-guide/introduction-to-dgxb300.html
# 稠密 BF16 和 HBM 标称峰值：https://inferencex.semianalysis.com/zh/chips/b300
# 单 GPU NVLink 互联标称 1.8 TB/s（双向合计）：https://www.nvidia.com/en-us/data-center/hgx/
HARDWARE = (
    {
        "name": "NVIDIA B300",
        "gpu_count": 8,
        "memory_bytes_per_gpu": 288_000_000_000,
        "peak_fp16_tflops": 2250,
        "peak_hbm_gb_s": 8000,
        "peak_interconnect_gb_s": 1800,
    },
)
