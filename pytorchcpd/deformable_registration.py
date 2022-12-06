from builtins import super
import torch
import numbers
from .emregistration import EMRegistration


def gaussian_kernel(X, beta, Y=None):
    """
    Calculate gaussian kernel matrix.

    Attributes
    ----------
    X: numpy array
        NxD array of points for creating gaussian.

    beta: float
        Width of the Gaussian kernel.

    Y: numpy array, optional
        MxD array of secondary points to calculate
        kernel with. Used if predicting on points
        not used to train.

    Returns
    -------
    K: numpy array
        Gaussian kernel matrix.
            NxN if Y is None
            NxM if Y is not None
    """
    if Y is None:
        Y = X
    diff = X[:, None, :] - Y[None, :,  :]
    diff = torch.square(diff)
    diff = torch.sum(diff, 2)
    return torch.exp(-diff / (2 * beta**2))

def low_rank_eigen(G, num_eig):
    """
    Calculate num_eig eigenvectors and eigenvalues of gaussian matrix G.
    Enables lower dimensional solving.

    Attributes
    ----------
    G: numpy array
        Gaussian kernel matrix.

    num_eig: int
        Number of eigenvectors to use in lowrank calculation.

    Returns
    -------
    Q: numpy array
        D x num_eig array of eigenvectors.

    S: numpy array
        num_eig array of eigenvalues.
    """
    S, Q = torch.linalg.eigh(G)
    eig_indices = list(torch.argsort(torch.abs(S))[::-1][:num_eig])
    Q = Q[:, eig_indices]  # eigenvectors
    S = S[eig_indices]  # eigenvalues.
    return Q, S


class DeformableRegistration(EMRegistration):
    """
    Deformable registration.

    Attributes
    ----------
    alpha: float (positive)
        Represents the trade-off between the goodness of maximum likelihood fit and regularization.

    beta: float(positive)
        Width of the Gaussian kernel.

    low_rank: bool
        Whether to use low rank approximation.

    num_eig: int
        Number of eigenvectors to use in lowrank calculation.
    """

    def __init__(self, alpha=None, beta=None, low_rank=False, num_eig=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if alpha is not None and (not isinstance(alpha, numbers.Number) or alpha <= 0):
            raise ValueError(
                "Expected a positive value for regularization parameter alpha. Instead got: {}".format(alpha))

        if beta is not None and (not isinstance(beta, numbers.Number) or beta <= 0):
            raise ValueError(
                "Expected a positive value for the width of the coherent Gaussian kerenl. Instead got: {}".format(beta))

        self.alpha = torch.tensor(2 if alpha is None else alpha, dtype=self.X.dtype, device=self.X.device)
        self.beta = torch.tensor(2 if beta is None else beta, dtype=self.X.dtype, device=self.X.device)
        self.W = torch.zeros((self.M, self.D), dtype=self.X.dtype, device=self.X.device)
        self.G = gaussian_kernel(self.Y, self.beta)
        self.low_rank = low_rank
        self.num_eig = num_eig
        if self.low_rank is True:
            self.Q, self.S = low_rank_eigen(self.G, self.num_eig)
            self.inv_S = torch.diag(1./self.S)
            self.S = torch.diag(self.S)
            self.E = 0.

    def update_transform(self):
        """
        Calculate a new estimate of the deformable transformation.
        See Eq. 22 of https://arxiv.org/pdf/0905.2635.pdf.

        """
        if self.low_rank is False:
            A = torch.matmul(torch.diag(self.P1), self.G) + \
                self.alpha * self.sigma2 * torch.eye(self.M, dtype=self.X.dtype, device=self.X.device)
            B = self.PX - torch.matmul(torch.diag(self.P1), self.Y)
            self.W = torch.linalg.solve(A, B)

        elif self.low_rank is True:
            # Matlab code equivalent can be found here:
            # https://github.com/markeroon/matlab-computer-vision-routines/tree/master/third_party/CoherentPointDrift
            dP = torch.diag(self.P1)
            dPQ = torch.matmul(dP, self.Q)
            F = self.PX - torch.matmul(dP, self.Y)

            self.W = 1 / (self.alpha * self.sigma2) * (F - torch.matmul(dPQ, (
                torch.linalg.solve((self.alpha * self.sigma2 * self.inv_S + torch.matmul(torch.t(self.Q), dPQ)),
                                (torch.matmul(torch.t(self.Q), F))))))
            QtW = torch.matmul(torch.t(self.Q), self.W)
            self.E = self.E + self.alpha / 2 * torch.trace(torch.matmul(torch.t(QtW), torch.matmul(self.S, QtW)))

    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the deformable transformation.

        Attributes
        ----------
        Y: numpy array, optional
            Array of points to transform - use to predict on new set of points.
            Best for predicting on new points not used to run initial registration.
                If None, self.Y used.

        Returns
        -------
        If Y is None, returns None.
        Otherwise, returns the transformed Y.


        """
        if Y is not None:
            G = gaussian_kernel(X=Y, beta=self.beta, Y=self.Y)
            return Y + torch.matmul(G, self.W)
        else:
            if self.low_rank is False:
                self.TY = self.Y + torch.matmul(self.G, self.W)

            elif self.low_rank is True:
                self.TY = self.Y + torch.matmul(self.Q, torch.matmul(self.S, torch.matmul(torch.t(self.Q), self.W)))
                return


    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the deformable transformation.
        See the update rule for sigma2 in Eq. 23 of of https://arxiv.org/pdf/0905.2635.pdf.

        """
        qprev = self.sigma2

        # The original CPD paper does not explicitly calculate the objective functional.
        # This functional will include terms from both the negative log-likelihood and
        # the Gaussian kernel used for regularization.
        self.q = torch.inf

        xPx = torch.matmul(torch.t(self.Pt1), torch.sum(
            torch.multiply(self.X, self.X), dim=1))
        yPy = torch.matmul(torch.t(self.P1),  torch.sum(
            torch.multiply(self.TY, self.TY), dim=1))
        trPXY = torch.sum(torch.multiply(self.TY, self.PX))

        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

        # Here we use the difference between the current and previous
        # estimate of the variance as a proxy to test for convergence.
        self.diff = torch.abs(self.sigma2 - qprev)

    def get_registration_parameters(self):
        """
        Return the current estimate of the deformable transformation parameters.


        Returns
        -------
        self.G: numpy array
            Gaussian kernel matrix.

        self.W: numpy array
            Deformable transformation matrix.
        """
        return self.G, self.W
