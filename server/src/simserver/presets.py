"""预置结构与标称规格；仿真用有效性能及驻留权重由使用者填写。"""

from simcore.schemas import ModelSpec

# Qwen 官方模型卡及 config.json：
# https://huggingface.co/Qwen/Qwen3-8B
# https://huggingface.co/Qwen/Qwen3-235B-A22B
MODELS = (
    ModelSpec(
        name="Qwen/Qwen3-8B", num_layers=36, hidden_size=4096,
        num_attention_heads=32, num_kv_heads=8, head_dim=128,
        ffn_intermediate_size=12288, total_parameters=8_200_000_000,
        vocab_size=151_936, tie_word_embeddings=False,
    ),
    ModelSpec(
        name="Qwen/Qwen3-235B-A22B", num_layers=94, hidden_size=4096,
        num_attention_heads=64, num_kv_heads=4, head_dim=128,
        ffn_intermediate_size=12288, expert_intermediate_size=1536,
        total_parameters=235_000_000_000, active_parameters=22_000_000_000,
        vocab_size=151_936, tie_word_embeddings=False,
        num_experts=128, experts_per_token=8, moe_layers=94,
    ),
)

# 单卡显存：https://docs.nvidia.com/dgx/dgxb300-user-guide/introduction-to-dgxb300.html
# 稠密 BF16 和 HBM 标称峰值：https://inferencex.semianalysis.com/zh/chips/b300
HARDWARE = (
    {
        "name": "NVIDIA B300", "gpu_count": 8, "memory_bytes_per_gpu": 288_000_000_000,
        "peak_fp16_tflops": 2250, "peak_hbm_gb_s": 8000,
    },
)
