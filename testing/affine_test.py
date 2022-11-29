import pytest
import torch
import numpy as np
from torch.testing import assert_close
from pycpd import AffineRegistration


def test_2D():
    B = torch.tensor([[1.0, 0.5], [0, 1.0]], dtype=torch.float64)
    t = torch.tensor([0.5, 1.0], dtype=torch.float64)

    Y = torch.from_numpy(np.loadtxt('data/fish_target.txt'))
    X = torch.matmul(Y, B) + torch.tile(t, (Y.shape[0], 1))

    reg = AffineRegistration(**{'X': X, 'Y': Y})
    TY, (B_reg, t_reg) = reg.register()
    assert_close(B, B_reg)
    assert_close(t, t_reg)
    assert_close(X, TY)


def test_3D():
    B = torch.tensor([[1.0, 0.5, 0.0], [0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    t = torch.tensor([0.5, 1.0, -2.0], dtype=torch.float64)

    fish_target = torch.from_numpy(np.loadtxt('data/fish_target.txt'))
    Y1 = torch.zeros((fish_target.shape[0], fish_target.shape[1] + 1), dtype=torch.float64)
    Y1[:, :-1] = fish_target
    Y2 = torch.ones((fish_target.shape[0], fish_target.shape[1] + 1), dtype=torch.float64)
    Y2[:, :-1] = fish_target
    Y = torch.vstack((Y1, Y2))

    X = torch.matmul(Y, B) + torch.tile(t, (Y.shape[0], 1))

    reg = AffineRegistration(**{'X': X, 'Y': Y})
    TY, (B_reg, t_reg) = reg.register()
    assert_close(B, B_reg)
    assert_close(t, t_reg)
    assert_close(X, TY)
