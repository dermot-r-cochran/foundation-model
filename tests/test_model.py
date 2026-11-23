"""Tests for FoundationModel."""

import math

import pytest
import torch

from foundation_model import FoundationModel, FoundationModelConfig, Trainer


@pytest.fixture
def tiny_config() -> FoundationModelConfig:
    return FoundationModelConfig.tiny()


@pytest.fixture
def tiny_model(tiny_config: FoundationModelConfig) -> FoundationModel:
    return FoundationModel(tiny_config)


def test_forward_shape(tiny_model: FoundationModel, tiny_config: FoundationModelConfig) -> None:
    B, T = 2, 8
    input_ids = torch.randint(0, tiny_config.vocab_size, (B, T))
    logits = tiny_model(input_ids)
    assert logits.shape == (B, T, tiny_config.vocab_size)


def test_predict_with_uncertainty_shapes(tiny_model: FoundationModel, tiny_config: FoundationModelConfig) -> None:
    B, T = 2, 8
    input_ids = torch.randint(0, tiny_config.vocab_size, (B, T))
    mean_logits, uncertainty = tiny_model.predict_with_uncertainty(input_ids, n_samples=5)
    assert mean_logits.shape == (B, T, tiny_config.vocab_size)
    assert uncertainty.shape == (B, T)
    assert (uncertainty >= 0).all()


def test_mcdropout_stays_on_in_eval() -> None:
    """MCDropout drops in eval(), where ordinary dropout is the identity.

    With p = 0.5 on 10,000 ones, every output is 0 or 2 (the 1 / (1 - p)
    rescale) and on seed 0 the dropped share is within 0.05 of a half; the
    same input through plain nn.Dropout in eval() comes back unchanged.
    This is the half that
    test_predict_with_uncertainty_is_higher_off_the_training_stream cannot
    separate from predict_with_uncertainty's own train() call.
    """
    from foundation_model.model import MCDropout

    torch.manual_seed(0)
    x = torch.ones(10_000)
    mc = MCDropout(0.5).eval()
    plain = torch.nn.Dropout(0.5).eval()

    out = mc(x)
    assert not mc.training
    assert set(out.unique().tolist()) == {0.0, 2.0}
    assert abs((out == 0).float().mean().item() - 0.5) < 0.05
    assert torch.equal(plain(x), x)


def test_model_forward_passes_differ_in_eval(
    tiny_model: FoundationModel, tiny_config: FoundationModelConfig
) -> None:
    """In eval() and without train(), two forward passes still differ.

    The model-level claim the README makes: every dropout in the model is
    an MCDropout, so a caller who puts the model in eval() still gets
    stochastic passes. With ordinary dropout the two passes would be equal.
    """
    torch.manual_seed(0)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
    tiny_model.eval()
    with torch.no_grad():
        first = tiny_model(ids)
        second = tiny_model(ids)
    assert not tiny_model.training
    assert not torch.equal(first, second)


def test_n_parameters_positive(tiny_model: FoundationModel) -> None:
    assert tiny_model.n_parameters > 0


@pytest.mark.parametrize("preset", ["tiny", "small", "base", "large"])
def test_all_presets_instantiate(preset: str) -> None:
    cfg = getattr(FoundationModelConfig, preset)()
    model = FoundationModel(cfg)
    assert model.n_parameters > 0


def test_weight_tying(tiny_model: FoundationModel) -> None:
    assert tiny_model.head.weight is tiny_model.token_embed.weight


# ------------------------------------------------------------------
# predict_with_uncertainty – what it returns, not only its shape
# ------------------------------------------------------------------


def test_predict_with_uncertainty_is_the_variance_of_the_softmax_across_passes(
    tiny_model: FoundationModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The epistemic figure is the variance across passes of each token's
    softmax probability, averaged over the vocabulary, and the first output
    is the mean of the passes' logits.

    Two scripted passes over a two-token vocabulary: logits [0, 0] give
    probabilities [0.5, 0.5], and [ln 3, 0] give [0.75, 0.25]. Each token's
    probability differs by 0.25 between the passes, so its variance (torch's
    default, unbiased: divided by S - 1 = 1) is 2 * 0.125**2 = 0.03125, and the
    mean over the two tokens is 0.03125. The mean logits are [ln 3 / 2, 0].

    Tolerance: float32 softmax and variance of order-1 values round at about
    1e-7, so 1e-6.

    It turns red if the measure returns zeros, uses the biased variance, takes
    the variance of the logits rather than the probabilities, or averages the
    logits wrongly.
    """
    passes = iter(
        [
            torch.tensor([[[0.0, 0.0]]]),
            torch.tensor([[[math.log(3.0), 0.0]]]),
        ]
    )
    monkeypatch.setattr(tiny_model, "forward", lambda input_ids: next(passes))

    mean_logits, uncertainty = tiny_model.predict_with_uncertainty(
        torch.zeros(1, 1, dtype=torch.long), n_samples=2
    )

    assert torch.allclose(
        mean_logits, torch.tensor([[[math.log(3.0) / 2, 0.0]]]), rtol=0.0, atol=1e-6
    )
    assert torch.allclose(uncertainty, torch.tensor([[0.03125]]), rtol=0.0, atol=1e-6)


def test_predict_with_uncertainty_is_zero_when_every_pass_agrees(
    tiny_config: FoundationModelConfig,
) -> None:
    """With no dropout every pass is the same forward pass, so the variance is
    zero at every position and the mean logits are the plain forward pass.

    Tolerance: the passes are the same computation on the same input, so
    they agree to the last bit on CPU; 1e-12 for the variance allows only
    rounding, and 1e-6 for the logits allows the mean of identical float32
    values to round.
    """
    tiny_config.dropout_rate = 0.0
    model = FoundationModel(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 8))

    mean_logits, uncertainty = model.predict_with_uncertainty(ids, n_samples=5)

    with torch.no_grad():
        plain = model(ids)
    assert torch.allclose(uncertainty, torch.zeros(2, 8), rtol=0.0, atol=1e-12)
    assert torch.allclose(mean_logits, plain, rtol=0.0, atol=1e-6)


def test_predict_with_uncertainty_is_higher_off_the_training_stream() -> None:
    """A model trained on a narrow stream gives a larger variance off it.

    The same setup as test_uncertainty.py::test_epistemic_higher_out_of_distribution,
    which measures the mutual information: the tiny model learns the cycle
    next = current + 1 mod 16 over a 256-token vocabulary for 200 steps, and
    tokens 16..255 are never seen. The model is put in eval() before the
    call, as a caller would; the passes still differ, but this test cannot
    say whether that is MCDropout or predict_with_uncertainty's own train()
    call, since either alone keeps dropout on (MCDropout in eval() is held
    on its own by test_mcdropout_stays_on_in_eval).

    Across 20 seeds on 2026-11-22 the ratio out/in of this variance measure
    was never below 4,672 (median about 6,000). The factor of 100 asserted
    leaves a margin of over 40 for a BLAS or torch change without being
    satisfiable by noise, and it fails if the measure returns zeros, since
    0 > 100 * 0 is false.
    """
    torch.manual_seed(0)
    config = FoundationModelConfig.tiny()
    config.vocab_size = 256
    config.learning_rate = 1e-3
    model = FoundationModel(config)
    trainer = Trainer(model, config)
    B, T, in_vocab = 8, 16, 16

    def in_distribution_batch():
        start = torch.randint(0, in_vocab, (B, 1))
        seq = (start + torch.arange(T + 1).unsqueeze(0)) % in_vocab
        return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()

    for _ in range(200):
        ids, labels = in_distribution_batch()
        trainer.train_step(ids, labels, store_in_buffer=False)

    model.eval()
    ids_in, _ = in_distribution_batch()
    ids_out = torch.randint(in_vocab, config.vocab_size, (B, T))
    _, v_in = model.predict_with_uncertainty(ids_in, n_samples=20)
    _, v_out = model.predict_with_uncertainty(ids_out, n_samples=20)
    assert v_out.mean() > 100 * v_in.mean(), (
        f"in-distribution {v_in.mean():.3e}, out {v_out.mean():.3e}"
    )
