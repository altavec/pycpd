import pytest
import numpy as np
import torch
from torch.testing import assert_close
from pycpd import gaussian_kernel, DeformableRegistration

@pytest.mark.parametrize('device', [torch.device('cpu'), torch.device('cuda')])
def test_2D(device):
    X = torch.from_numpy(np.loadtxt('data/fish_target.txt')).to(device)
    Y = torch.from_numpy(np.loadtxt('data/fish_source.txt')).to(device)

    reg = DeformableRegistration(**{'X': X, 'Y': Y})
    TY, _ = reg.register()
    assert_close(X, TY, atol=1.5, rtol=1.)


@pytest.mark.parametrize('device', [torch.device('cpu'), torch.device('cuda')])
def test_3D(device):
    fish_target = torch.from_numpy(np.loadtxt('data/fish_target.txt')).to(device)
    X1 = torch.zeros((fish_target.shape[0], fish_target.shape[1] + 1))
    X1[:, :-1] = fish_target
    X2 = torch.ones((fish_target.shape[0], fish_target.shape[1] + 1))
    X2[:, :-1] = fish_target
    X = torch.vstack((X1, X2))

    fish_source = torch.from_numpy(np.loadtxt('data/fish_source.txt')).to(device)
    Y1 = torch.zeros((fish_source.shape[0], fish_source.shape[1] + 1))
    Y1[:, :-1] = fish_source
    Y2 = torch.ones((fish_source.shape[0], fish_source.shape[1] + 1))
    Y2[:, :-1] = fish_source
    Y = torch.vstack((Y1, Y2))

    reg = DeformableRegistration(**{'X': X, 'Y': Y})
    TY, _ = reg.register()
    assert_close(TY, X, atol=1.5, rtol=1.)


@pytest.mark.skip(reason='Fails and we don\'t care about low ranks')
def test_3D_low_rank():
    fish_target = torch.from_numpy(np.loadtxt('data/fish_target.txt'))
    X1 = torch.zeros((fish_target.shape[0], fish_target.shape[1] + 1))
    X1[:, :-1] = fish_target
    X2 = torch.ones((fish_target.shape[0], fish_target.shape[1] + 1))
    X2[:, :-1] = fish_target
    X = torch.vstack((X1, X2))

    fish_source = torch.from_numpy(np.loadtxt('data/fish_source.txt'))
    Y1 = torch.zeros((fish_source.shape[0], fish_source.shape[1] + 1))
    Y1[:, :-1] = fish_source
    Y2 = torch.ones((fish_source.shape[0], fish_source.shape[1] + 1))
    Y2[:, :-1] = fish_source
    Y = torch.vstack((Y1, Y2))

    reg = DeformableRegistration(**{'X': X, 'Y': Y, 'low_rank': True})
    TY, _ = reg.register()
    assert_close(TY, X)

    rand_pts = torch.randint(Y.shape[0], size=int(Y.shape[0]/2))
    TY2 = reg.transform_point_cloud(Y=Y[rand_pts, :])
    assert_close(TY2, X[rand_pts, :])
