import pytest
import numpy as np
import torch
from torch.testing import assert_close
from pycpd import RigidRegistration


def test_2D():
    theta = torch.tensor(torch.pi / 6.0, dtype=torch.float64)
    R = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                  [torch.sin(theta), torch.cos(theta)]], dtype=torch.float64)
    t = torch.tensor([0.5, 1.0], dtype=torch.float64)

    Y = torch.from_numpy(np.loadtxt('data/fish_target.txt'))
    X = torch.matmul(Y, R) + torch.tile(t, (Y.shape[0], 1))

    reg = RigidRegistration(**{'X': X, 'Y': Y})
    TY, (s_reg, R_reg, t_reg) = reg.register()
    assert_close(torch.tensor(1, dtype=torch.float64), s_reg)
    assert_close(R, R_reg)
    assert_close(t, t_reg)
    assert_close(X, TY)


def test_3D():
    theta = torch.tensor(torch.pi / 6.0, dtype=torch.float64)
    R = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                  [torch.sin(theta), torch.cos(theta), 0], [0, 0, 1]], dtype=torch.float64)
    t = torch.tensor([0.5, 1.0, -2.0], dtype=torch.float64)

    fish_target = torch.from_numpy(np.loadtxt('data/fish_target.txt'))
    Y = torch.zeros((fish_target.shape[0], fish_target.shape[1] + 1), dtype=torch.float64)
    Y[:, :-1] = fish_target
    X = torch.matmul(Y, R) + torch.tile(t, (Y.shape[0], 1))

    reg = RigidRegistration(**{'X': X, 'Y': Y})
    TY, (s_reg, R_reg, t_reg) = reg.register()
    assert_close(torch.tensor(1, dtype=torch.float64), s_reg)
    assert_close(R, R_reg)
    assert_close(t, t_reg)
    assert_close(X, TY)
