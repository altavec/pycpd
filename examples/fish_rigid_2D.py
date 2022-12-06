from functools import partial
import matplotlib.pyplot as plt
from pytorchcpd import RigidRegistration
import torch
import numpy as np


def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Source')
    plt.text(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def main(true_rigid=True):
    X = torch.from_numpy(np.loadtxt('data/fish_target.txt'))
    if true_rigid is True:
        theta = torch.tensor(torch.pi / 6.0, dtype=torch.float64)
        R = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]])
        t = torch.tensor([0.5, 1.0])
        Y = torch.matmul(X, R) + t
    else:
        Y = torch.from_numpy(np.loadtxt('data/fish_source.txt'))

    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    callback = partial(visualize, ax=fig.axes[0])

    reg = RigidRegistration(**{'X': X, 'Y': Y})
    reg.register(callback)
    plt.show()


if __name__ == '__main__':
    main(true_rigid=True)
