import copy

import pytest

from open_mythos import MythosConfig


@pytest.fixture
def tiny_config_gqa():
    return MythosConfig(
        vocab_size=512,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=128,
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=32,
        lora_rank=4,
        attn_type="gqa",
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
    )


@pytest.fixture
def tiny_config_mla(tiny_config_gqa):
    cfg = copy.deepcopy(tiny_config_gqa)
    cfg.attn_type = "mla"
    return cfg
