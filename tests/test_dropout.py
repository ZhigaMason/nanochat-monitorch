"""
Test attention-probability dropout and MLP hidden dropout.

Run: python -m pytest tests/test_dropout.py -v

These tests exercise the SDPA path only. FA3 does not support dropout, so
CausalSelfAttention is expected to refuse to run with attn_dropout > 0 on a
Hopper GPU (see test_fa3_rejects_dropout).
"""
import pytest
import torch

import nanochat.flash_attention as fa_module
from nanochat.flash_attention import flash_attn, HAS_FA3
from nanochat.common import COMPUTE_DTYPE
from nanochat.gpt import GPT, GPTConfig

DEVICE = "cpu"


def set_impl(impl):
    """Set the implementation override ('fa3', 'sdpa', or None for auto) and re-resolve USE_FA3."""
    fa_module._override_impl = impl
    fa_module.USE_FA3 = fa_module._resolve_use_fa3()


@pytest.fixture(autouse=True)
def force_sdpa():
    """All tests here run on the SDPA path; reset the override afterwards."""
    set_impl('sdpa')
    yield
    set_impl(None)


def build_tiny_gpt(**config_kwargs):
    """A minimal GPT that can be built and run on CPU.

    init_weights() zero-initializes attn.c_proj and mlp.c_proj, which makes every
    sublayer contribute exactly zero to the residual stream at init - dropout inside
    those sublayers could then never move the logits. Give both projections real
    weights so the tests actually observe the sublayer output.
    """
    config = GPTConfig(
        sequence_len=32, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2,
        n_embd=64, window_pattern="L", **config_kwargs,
    )
    model = GPT(config)
    model.init_weights()
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.02)
            block.mlp.c_proj.weight.normal_(std=0.02)
    return model


# =============================================================================
# flash_attn_func plumbing
# =============================================================================
class TestAttentionDropoutPlumbing:

    def test_dropout_p_changes_output(self):
        """dropout_p > 0 makes flash_attn_func stochastic on the SDPA path."""
        B, T, H, D = 2, 16, 2, 16
        q = torch.randn(B, T, H, D, device=DEVICE)
        k = torch.randn(B, T, H, D, device=DEVICE)
        v = torch.randn(B, T, H, D, device=DEVICE)

        torch.manual_seed(0)
        y1 = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1), dropout_p=0.5)
        torch.manual_seed(1)
        y2 = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1), dropout_p=0.5)

        assert not torch.equal(y1, y2), "dropout_p=0.5 produced identical outputs across RNG states"

    def test_dropout_p_zero_matches_no_dropout(self):
        """dropout_p=0.0 is bit-identical to omitting the argument (regression guard)."""
        B, T, H, D = 2, 16, 2, 16
        q = torch.randn(B, T, H, D, device=DEVICE)
        k = torch.randn(B, T, H, D, device=DEVICE)
        v = torch.randn(B, T, H, D, device=DEVICE)

        y_default = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))
        y_explicit = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1), dropout_p=0.0)

        assert torch.equal(y_default, y_explicit)

    def test_sliding_window_supports_dropout(self):
        """The sliding-window SDPA branch also honours dropout_p."""
        B, T, H, D = 2, 32, 2, 16
        q = torch.randn(B, T, H, D, device=DEVICE)
        k = torch.randn(B, T, H, D, device=DEVICE)
        v = torch.randn(B, T, H, D, device=DEVICE)

        torch.manual_seed(0)
        y1 = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(8, 0), dropout_p=0.5)
        torch.manual_seed(1)
        y2 = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(8, 0), dropout_p=0.5)

        assert not torch.equal(y1, y2), "sliding-window branch ignored dropout_p"

    @pytest.mark.skipif(not HAS_FA3, reason="requires a Hopper GPU to exercise the FA3 path")
    def test_fa3_rejects_dropout(self):
        """FA3 has no dropout support, so it must fail loudly rather than silently skip it."""
        set_impl('fa3')
        B, T, H, D = 2, 16, 2, 16
        q = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16)

        with pytest.raises(AssertionError, match="dropout"):
            flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1), dropout_p=0.1)


# =============================================================================
# Model-level behaviour
# =============================================================================
class TestModelDropout:

    def test_defaults_are_off(self):
        """Both rates default to 0.0 so existing runs are unaffected."""
        config = GPTConfig()
        assert config.attn_dropout == 0.0
        assert config.mlp_dropout == 0.0

    def test_mlp_dropout_is_stochastic_in_train_mode(self):
        """mlp_dropout > 0 makes repeated training forward passes differ."""
        torch.manual_seed(0)
        model = build_tiny_gpt(mlp_dropout=0.5)
        model.train()
        idx = torch.randint(0, 64, (2, 16), device=DEVICE)

        torch.manual_seed(0)
        logits1 = model(idx)
        torch.manual_seed(1)
        logits2 = model(idx)

        assert not torch.equal(logits1, logits2), "mlp_dropout had no effect in train mode"

    def test_attn_dropout_is_stochastic_in_train_mode(self):
        """attn_dropout > 0 makes repeated training forward passes differ."""
        torch.manual_seed(0)
        model = build_tiny_gpt(attn_dropout=0.5)
        model.train()
        idx = torch.randint(0, 64, (2, 16), device=DEVICE)

        torch.manual_seed(0)
        logits1 = model(idx)
        torch.manual_seed(1)
        logits2 = model(idx)

        assert not torch.equal(logits1, logits2), "attn_dropout had no effect in train mode"

    def test_dropout_disabled_in_eval_mode(self):
        """Both dropouts must be inert under eval(), so validation loss is deterministic."""
        torch.manual_seed(0)
        model = build_tiny_gpt(attn_dropout=0.5, mlp_dropout=0.5)
        model.eval()
        idx = torch.randint(0, 64, (2, 16), device=DEVICE)

        torch.manual_seed(0)
        logits1 = model(idx)
        torch.manual_seed(1)
        logits2 = model(idx)

        assert torch.equal(logits1, logits2), "dropout leaked into eval mode"

    def test_zero_rates_match_baseline(self):
        """A model with rates at 0.0 is bit-identical to one built without the fields."""
        torch.manual_seed(0)
        baseline = build_tiny_gpt()
        torch.manual_seed(0)
        explicit_zero = build_tiny_gpt(attn_dropout=0.0, mlp_dropout=0.0)
        idx = torch.randint(0, 64, (2, 16), device=DEVICE)

        baseline.train()
        explicit_zero.train()
        torch.manual_seed(0)
        logits_baseline = baseline(idx)
        torch.manual_seed(0)
        logits_zero = explicit_zero(idx)

        assert torch.equal(logits_baseline, logits_zero)


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"COMPUTE_DTYPE: {COMPUTE_DTYPE}")
    print(f"HAS_FA3: {HAS_FA3}")
    pytest.main([__file__, "-v", "-s"])
