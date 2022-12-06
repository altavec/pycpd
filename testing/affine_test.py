import pytest
import torch
import numpy as np
from torch.testing import assert_close
from pytorchcpd import AffineRegistration


@pytest.mark.parametrize('device', [torch.device('cpu'), torch.device('cuda')])
def test_2D(device):
    B = torch.tensor([[1.0, 0.5], [0, 1.0]], dtype=torch.float64, device=device)
    t = torch.tensor([0.5, 1.0], dtype=torch.float64, device=device)

    Y = torch.from_numpy(np.loadtxt('data/fish_target.txt')).to(device)
    X = torch.matmul(Y, B) + torch.tile(t, (Y.shape[0], 1))

    reg = AffineRegistration(**{'X': X, 'Y': Y})
    TY, (B_reg, t_reg) = reg.register()
    assert_close(B, B_reg)
    assert_close(t, t_reg)
    assert_close(X, TY)


@pytest.mark.parametrize('device', [torch.device('cpu'), torch.device('cuda')])
def test_3D(device):
    B = torch.tensor([[1.0, 0.5, 0.0], [0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64, device=device)
    t = torch.tensor([0.5, 1.0, -2.0], dtype=torch.float64, device=device)

    fish_target = torch.from_numpy(np.loadtxt('data/fish_target.txt')).to(device)
    Y1 = torch.zeros((fish_target.shape[0], fish_target.shape[1] + 1), dtype=torch.float64, device=device)
    Y1[:, :-1] = fish_target
    Y2 = torch.ones((fish_target.shape[0], fish_target.shape[1] + 1), dtype=torch.float64, device=device)
    Y2[:, :-1] = fish_target
    Y = torch.vstack((Y1, Y2))

    X = torch.matmul(Y, B) + torch.tile(t, (Y.shape[0], 1))

    reg = AffineRegistration(**{'X': X, 'Y': Y})
    TY, (B_reg, t_reg) = reg.register()
    assert_close(B, B_reg)
    assert_close(t, t_reg)
    assert_close(X, TY)
