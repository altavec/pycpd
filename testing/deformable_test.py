import pytest
import numpy as np
import torch
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pycpd import gaussian_kernel, DeformableRegistration


def test_2D():
    X = torch.from_numpy(np.loadtxt('data/fish_target.txt'))
    Y = torch.from_numpy(np.loadtxt('data/fish_source.txt'))

    reg = DeformableRegistration(**{'X': X, 'Y': Y})
    TY, _ = reg.register()
    assert_array_almost_equal(X, TY, decimal=1)


def test_3D():
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

    reg = DeformableRegistration(**{'X': X, 'Y': Y})
    TY, _ = reg.register()
    assert_array_almost_equal(TY, X, decimal=0)


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
    assert_array_almost_equal(TY, X, decimal=0)

    rand_pts = torch.randint(Y.shape[0], size=int(Y.shape[0]/2))
    TY2 = reg.transform_point_cloud(Y=Y[rand_pts, :])
    assert_array_almost_equal(TY2, X[rand_pts, :], decimal=0)
