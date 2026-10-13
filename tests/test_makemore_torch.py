"""Tests for the PyTorch notebooks. Skipped automatically when torch is not installed."""
import math
import os

import pytest

from conftest import ROOT, cell_containing

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def bigram(words):
    """Run the counting cells of BigramMakemore.ipynb on the first 500 names (fast, no network)."""
    os.chdir(ROOT)
    ns = {"torch": torch, "words": words}
    exec("N = torch.zeros((27,27), dtype=torch.int32)", ns)
    exec(cell_containing("BigramMakemore", "stoi['.'] = 0"), ns)   # vocabulary from all names
    ns["words"] = words[:500]                                      # count only a subset, for speed
    exec(cell_containing("BigramMakemore", "N[ix1, ix2] += 1"), ns)
    ns["N"] += 1                          # the notebook's +1 smoothing (cell 11)
    P = ns["N"].float()
    P /= P.sum(1, keepdim=True)
    ns["P"] = P
    return ns


def test_smoothing_leaves_no_zero_counts(bigram):
    assert bigram["N"].min().item() >= 1


def test_rows_of_P_are_probability_distributions(bigram):
    assert torch.allclose(bigram["P"].sum(1), torch.ones(27), atol=1e-5)


def test_stoi_has_dot_at_zero_and_27_symbols(bigram):
    assert bigram["stoi"]["."] == 0 and len(bigram["stoi"]) == 27


def test_softmax_cells_equal_torch_softmax():
    """Notebook's exp/normalise lines are a softmax."""
    logits = torch.randn(5, 27, generator=torch.Generator().manual_seed(0))
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    assert torch.allclose(probs, torch.softmax(logits, 1), atol=1e-6)


def test_uniform_model_loss_is_log_27():
    probs = torch.full((5, 27), 1 / 27)
    ys = torch.tensor([5, 13, 13, 1, 0])
    loss = -probs[torch.arange(5), ys].log().mean()
    assert loss.item() == pytest.approx(math.log(27), rel=1e-5)


def test_one_layer_net_loss_decreases():
    """Tiny version of 1LayerNNMakemore cell 45: gradient descent lowers the loss."""
    import torch.nn.functional as F
    xs = torch.tensor([0, 5, 13, 13, 1])
    ys = torch.tensor([5, 13, 13, 1, 0])
    W = torch.randn((27, 27), generator=torch.Generator().manual_seed(2147483647), requires_grad=True)
    losses = []
    for _ in range(20):
        logits = F.one_hot(xs, num_classes=27).float() @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)
        loss = -probs[torch.arange(5), ys].log().mean()
        W.grad = None
        loss.backward()
        W.data += -1.0 * W.grad
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_mlp_context_window_shapes(words):
    """MLPMakemore build_dataset: one (3-char context, next char) row per character incl. the end marker."""
    ns = {"torch": torch}
    ns["words"] = words
    exec("chars = sorted(list(set(''.join(words))))\nstoi = {s:i+1 for i,s in enumerate(chars)}\nstoi['.'] = 0", ns)
    exec(cell_containing("MLPMakemore", "def build_dataset"). split("import random")[0], ns)
    X, Y = ns["build_dataset"](["emma"])
    assert X.shape == (5, 3) and Y.shape == (5,)
    assert X[0].tolist() == [0, 0, 0]
    assert X[1].tolist() == [0, 0, ns["stoi"]["e"]]
    assert Y[-1].item() == 0
