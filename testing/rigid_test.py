import pytest
import numpy as np
import torch
from torch.testing import assert_close
from pytorchcpd import RigidRegistration


@pytest.mark.parametrize('device', [torch.device('cpu'), torch.device('cuda')])
def test_2D(device):
    theta = torch.tensor(torch.pi / 6.0, dtype=torch.float64, device=device)
    R = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                  [torch.sin(theta), torch.cos(theta)]], dtype=torch.float64, device=device)
    t = torch.tensor([0.5, 1.0], dtype=torch.float64, device=device)

    Y = torch.from_numpy(np.loadtxt('data/fish_target.txt')).to(device)
    X = torch.matmul(Y, R) + torch.tile(t, (Y.shape[0], 1))

    reg = RigidRegistration(**{'X': X, 'Y': Y})
    TY, (s_reg, R_reg, t_reg) = reg.register()
    assert_close(torch.tensor(1, dtype=torch.float64, device=device), s_reg)
    assert_close(R, R_reg)
    assert_close(t, t_reg)
    assert_close(X, TY)


@pytest.mark.parametrize('device', [torch.device('cpu'), torch.device('cuda')])
def test_3D(device):
    theta = torch.tensor(torch.pi / 6.0, dtype=torch.float64, device=device)
    R = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                  [torch.sin(theta), torch.cos(theta), 0], [0, 0, 1]], dtype=torch.float64, device=device)
    t = torch.tensor([0.5, 1.0, -2.0], dtype=torch.float64, device=device)

    fish_target = torch.from_numpy(np.loadtxt('data/fish_target.txt')).to(device)
    Y = torch.zeros((fish_target.shape[0], fish_target.shape[1] + 1), dtype=torch.float64, device=device)
    Y[:, :-1] = fish_target
    X = torch.matmul(Y, R) + torch.tile(t, (Y.shape[0], 1))

    reg = RigidRegistration(**{'X': X, 'Y': Y})
    TY, (s_reg, R_reg, t_reg) = reg.register()
    assert_close(torch.tensor(1, dtype=torch.float64, device=device), s_reg)
    assert_close(R, R_reg)
    assert_close(t, t_reg)
    assert_close(X, TY)
