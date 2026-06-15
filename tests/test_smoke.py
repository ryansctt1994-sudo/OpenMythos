import torch
import pytest

from open_mythos import OpenMythos


@pytest.mark.parametrize("config_fixture", ["tiny_config_gqa", "tiny_config_mla"])
def test_forward_backward_smoke(config_fixture, request):
    cfg = request.getfixturevalue(config_fixture)
    torch.manual_seed(42)
    model = OpenMythos(cfg)

    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(ids, n_loops=2)

    assert logits.shape == (2, 16, cfg.vocab_size), f"Invalid logit shape: {logits.shape}"
    assert torch.isfinite(logits).all(), "NaN or Inf detected in forward pass"

    loss = logits.mean()
    loss.backward()

    has_grad = any(p.grad is not None for p in model.prelude.parameters())
    assert has_grad, "Gradient flow severed between loss and Prelude layer."


def test_generate_smoke(tiny_config_gqa):
    torch.manual_seed(42)
    model = OpenMythos(tiny_config_gqa)
    ids = torch.randint(0, tiny_config_gqa.vocab_size, (1, 5))

    out = model.generate(ids, max_new_tokens=4, n_loops=2)

    assert out.shape[0] == 1
    assert out.shape[1] <= 9, "Generation failed to halt within token bounds."
