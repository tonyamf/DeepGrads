import math
import random

import pytest


def test_forward_arithmetic(micrograd):
    V = micrograd["Value"]
    assert (V(2.0) * V(-3.0) + V(10.0)).data == 4.0
    assert (V(2) - V(5)).data == -3
    assert (3 - V(2)).data == 1          # __rsub__
    assert (V(6) / V(3)).data == 2.0
    assert (2 * V(3)).data == 6          # __rmul__
    assert (V(2) ** 3).data == 8


def test_backward_add_mul_matches_hand_derivative(micrograd):
    V = micrograd["Value"]
    a, b, c = V(2.0), V(-3.0), V(10.0)
    d = a * b + c
    d.backward()
    assert (a.grad, b.grad, c.grad) == (-3.0, 2.0, 1.0)


def test_gradient_accumulates_when_a_node_is_used_twice(micrograd):
    V = micrograd["Value"]
    a = V(3.0)
    (a * a).backward()
    assert a.grad == 6.0                 # d(a^2)/da = 2a


def test_pow_and_div_gradients(micrograd):
    V = micrograd["Value"]
    a, b = V(3.0), V(2.0)
    y = a ** 3 + b / a - 1
    y.backward()
    assert a.grad == pytest.approx(3 * 9 - 2 / 9)
    assert b.grad == pytest.approx(1 / 3)


def test_tanh_gradient_matches_finite_difference(micrograd):
    V = micrograd["Value"]
    x = V(0.7)
    y = x.tanh()
    y.backward()
    h = 1e-6
    numeric = (math.tanh(0.7 + h) - math.tanh(0.7 - h)) / (2 * h)
    assert x.grad == pytest.approx(numeric, rel=1e-6)


def test_karpathy_neuron_example(micrograd):
    """The notebook's x1,x2,w1,w2,b example: o = tanh(...) ~ 0.7071."""
    V = micrograd["Value"]
    x1, x2, w1, w2 = V(2.0), V(0.0), V(-3.0), V(1.0)
    b = V(6.8813735870195432)
    o = (x1 * w1 + x2 * w2 + b).tanh()
    o.backward()
    assert o.data == pytest.approx(0.7071, abs=1e-4)
    assert x1.grad == pytest.approx(-1.5, abs=1e-3)
    assert w1.grad == pytest.approx(1.0, abs=1e-3)


def test_mlp_shape_and_parameter_count(micrograd):
    random.seed(0)
    mlp = micrograd["MLP"](3, [4, 4, 1])
    assert len(mlp.parameters()) == 41   # (3+1)*4 + (4+1)*4 + (4+1)*1
    out = mlp([2.0, 3.0, -1.0])
    assert isinstance(out, micrograd["Value"])
    assert -1.0 < out.data < 1.0         # last layer is tanh


def test_training_loop_reduces_loss(micrograd):
    """Same four points and learning rate as the notebook's training cell."""
    random.seed(1)
    mlp = micrograd["MLP"](3, [4, 4, 1])
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]
    losses = []
    for _ in range(20):
        loss = sum((mlp(x) - y) ** 2 for x, y in zip(xs, ys))
        for p in mlp.parameters():
            p.grad = 0.0
        loss.backward()
        for p in mlp.parameters():
            p.data += -0.1 * p.grad
        losses.append(loss.data)
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.5


@pytest.mark.xfail(strict=True, reason="BUG micrograd cell 8: Value.exp._backward uses '=' not '+=', "
                                       "so a node used twice loses gradient (see 2_fix/review.md)")
def test_exp_gradient_accumulates(micrograd):
    V = micrograd["Value"]
    a = V(1.0)
    (a.exp() * a).backward()             # d/da (e^a * a) = 2e at a=1
    assert a.grad == pytest.approx(2 * math.e)


@pytest.mark.xfail(strict=True, raises=OverflowError,
                   reason="BUG micrograd cell 8: tanh uses math.exp(2n), overflows for n > ~355")
def test_tanh_large_input_is_finite(micrograd):
    assert micrograd["Value"](400.0).tanh().data == pytest.approx(1.0)
