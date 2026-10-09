"""仅提供结构/标称容量的目录；有效性能与驻留权重由使用者填写。"""

from simcore.schemas import ModelSpec

MODELS = (
    ModelSpec(
        name="Qwen/Qwen3-8B", num_layers=36, hidden_size=4096,
        num_attention_heads=32, num_kv_heads=8, head_dim=128,
        ffn_intermediate_size=12288, total_parameters=8_200_000_000,
    ),
    ModelSpec(
        name="Qwen/Qwen3-235B-A22B", num_layers=94, hidden_size=4096,
        num_attention_heads=64, num_kv_heads=4, head_dim=128,
        ffn_intermediate_size=12288, expert_intermediate_size=1536,
        total_parameters=235_000_000_000, active_parameters=22_000_000_000,
        num_experts=128, experts_per_token=8, moe_layers=94,
    ),
)

HARDWARE = (
    {"name": "NVIDIA B300", "gpu_count": 8, "memory_bytes_per_gpu": 288_000_000_000},
)
