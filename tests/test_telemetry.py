import pytest
import torch

from open_mythos.telemetry import (
    TelemetryState,
    active_governance_gate,
    compute_expert_telemetry,
    get_current_telemetry,
    observe_telemetry,
    spectral_radius_from_A,
)


def test_observe_telemetry_restores_previous_context():
    outer = TelemetryState()
    inner = TelemetryState()

    assert get_current_telemetry() is None

    with observe_telemetry(outer) as active_outer:
        assert active_outer is outer
        assert get_current_telemetry() is outer

        with observe_telemetry(inner) as active_inner:
            assert active_inner is inner
            assert get_current_telemetry() is inner

        assert get_current_telemetry() is outer

    assert get_current_telemetry() is None


def test_compute_expert_telemetry_balanced_counts():
    metrics = compute_expert_telemetry(torch.tensor([10, 10, 10, 10]))

    assert metrics["dead_experts"] == 0
    assert metrics["expert_gini"] == pytest.approx(0.0)
    assert metrics["expert_entropy_normalized"] == pytest.approx(1.0)
    assert metrics["total_routings"] == 40


def test_spectral_radius_vector_fast_path():
    A = torch.tensor([0.1, -0.5, 0.25])

    assert spectral_radius_from_A(A) == pytest.approx(0.5)


def test_active_governance_gate_raises_on_bad_metrics():
    with pytest.raises(RuntimeError):
        active_governance_gate(
            {
                "loss": 1.0,
                "spectral_radius": 1.10,
                "dead_experts": 0,
                "expert_gini": 0.1,
            }
        )
