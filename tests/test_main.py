"""Tests for the demo entry point (``python -m foundation_model``).

This is what the Docker ``cpu`` and ``gpu`` targets run, selected by the
``DEMO_MODE`` environment variable that ``docker-compose.yml`` sets. Each
test runs one mode in-process on CPU and checks the demo's own headings, so
a demo that raises or silently skips a section fails here before it fails
in a container.
"""

import pytest

from foundation_model.__main__ import main


def _run(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, mode: str) -> str:
    monkeypatch.setenv("DEMO_MODE", mode)
    monkeypatch.setenv("DEVICE", "cpu")
    main()
    return capsys.readouterr().out


def test_single_mode_reports_both_uncertainties(monkeypatch, capsys) -> None:
    out = _run(monkeypatch, capsys, "single")
    assert "mode=single" in out
    assert "Single-Node Epistemic Inference" in out
    assert "Epistemic" in out and "Aleatoric" in out
    assert "Federated Learning Demo" not in out
    assert "P2P Gossip Demo" not in out
    assert out.rstrip().endswith("Done")


def test_federated_mode_runs_both_strategies(monkeypatch, capsys) -> None:
    out = _run(monkeypatch, capsys, "federated")
    assert "Federated Learning Demo" in out
    assert "[   fedavg] round 2 complete" in out
    assert "[expertise] round 2 complete" in out
    assert "Single-Node" not in out


def test_p2p_mode_runs_uniform_and_expertise(monkeypatch, capsys) -> None:
    out = _run(monkeypatch, capsys, "p2p")
    assert "P2P Gossip Demo" in out
    assert "[uniform] 10 gossip rounds over 4 peers" in out
    assert "[expertise-weighted] 10 gossip rounds over 4 peers" in out
    assert "Single-Node" not in out


def test_all_mode_is_the_default_and_runs_every_demo(monkeypatch, capsys) -> None:
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setenv("DEVICE", "cpu")
    main()
    out = capsys.readouterr().out
    assert "mode=all" in out
    for heading in ("Single-Node Epistemic Inference", "Federated Learning Demo", "P2P Gossip Demo"):
        assert heading in out


def test_cuda_request_falls_back_to_cpu_when_unavailable(monkeypatch, capsys) -> None:
    """DEVICE=cuda on a CPU-only host must not raise; the demo runs on cpu."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.setenv("DEMO_MODE", "single")
    main()
    out = capsys.readouterr().out
    assert "device=cpu" in out
