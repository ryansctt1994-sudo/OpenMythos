import copy
import os

import torch

from open_mythos import OpenMythos


# Required by CUDA deterministic matmul paths when these tests are executed on GPU.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def set_strict_determinism(seed: int = 42) -> None:
    """Enforce strict reproducibility for local replay tests."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=False)


def test_byte_for_byte_replay(tiny_config_gqa):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_strict_determinism(42)
    base_model = OpenMythos(tiny_config_gqa).to(device)
    test_ids = torch.randint(0, tiny_config_gqa.vocab_size, (2, 16), device=device)

    set_strict_determinism(1337)
    model_1 = copy.deepcopy(base_model)
    out_1 = model_1(test_ids, n_loops=2)
    loss_1 = out_1.mean()
    loss_1.backward()

    out_1_saved = out_1.detach().cpu().clone()
    loss_1_saved = loss_1.detach().cpu().clone()
    grad_1 = {
        name: p.grad.detach().cpu().clone()
        for name, p in model_1.named_parameters()
        if p.grad is not None
    }

    del model_1, out_1, loss_1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    set_strict_determinism(1337)
    model_2 = copy.deepcopy(base_model)
    out_2 = model_2(test_ids, n_loops=2)
    loss_2 = out_2.mean()
    loss_2.backward()

    assert torch.equal(out_1_saved, out_2.detach().cpu()), "Forward pass non-determinism."
    assert loss_1_saved.item() == loss_2.item(), "Loss scalar non-determinism."

    grad_2 = {
        name: p.grad.detach().cpu()
        for name, p in model_2.named_parameters()
        if p.grad is not None
    }
    assert grad_1.keys() == grad_2.keys(), "Gradient parameter set changed between replays."

    for name in grad_1.keys():
        assert torch.equal(grad_1[name], grad_2[name]), (
            f"Gradient non-determinism in layer: {name}"
        )
