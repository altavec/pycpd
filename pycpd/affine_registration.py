from builtins import super
import torch
from .emregistration import EMRegistration
from .utility import is_positive_semi_definite


class AffineRegistration(EMRegistration):
    """
    Affine registration.

    Attributes
    ----------
    B: numpy array (semi-positive definite)
        DxD affine transformation matrix.

    t: numpy array
        1xD initial translation vector.
    """
    # Additional parameters used in this class, but not inputs.
    # YPY: float
    #     Denominator value used to update the scale factor.
    #     Defined in Fig. 2 and Eq. 8 of https://arxiv.org/pdf/0905.2635.pdf.

    # X_hat: numpy array
    #     Centered target point cloud.
    #     Defined in Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf

    def __init__(self, B=None, t=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if B is not None and ((B.ndim != 2) or (B.shape[0] != self.D) or (B.shape[1] != self.D) or not is_positive_semi_definite(B)):
            raise ValueError(
                'The rotation matrix can only be initialized to {}x{} positive semi definite matrices. Instead got: {}.'.format(self.D, self.D, B))

        if t is not None and ((t.ndim != 2) or (t.shape[0] != 1) or (t.shape[1] != self.D)):
            raise ValueError(
                'The translation vector can only be initialized to 1x{} positive semi definite matrices. Instead got: {}.'.format(self.D, t))
        self.B = torch.eye(self.D, dtype=self.X.dtype, device=self.X.device) if B is None else B
        self.t = torch.atleast_2d(torch.zeros((1, self.D), dtype=self.X.dtype, device=self.X.device)) if t is None else t

        self.YPY = None
        self.X_hat = None
        self.A = None

    def update_transform(self):
        """
        Calculate a new estimate of the rigid transformation.

        """

        # source and target point cloud means
        muX = torch.divide(torch.sum(self.PX, dim=0), self.Np)
        muY = torch.divide(
            torch.sum(torch.matmul(torch.t(self.P), self.Y), dim=0), self.Np)

        self.X_hat = self.X - torch.tile(muX, (self.N, 1))
        Y_hat = self.Y - torch.tile(muY, (self.M, 1))

        self.A = torch.matmul(torch.t(self.X_hat), torch.t(self.P))
        self.A = torch.matmul(self.A, Y_hat)

        self.YPY = torch.matmul(torch.t(Y_hat), torch.diag(self.P1))
        self.YPY = torch.matmul(self.YPY, Y_hat)

        # Calculate the new estimate of affine parameters using update rules for (B, t)
        # as defined in Fig. 3 of https://arxiv.org/pdf/0905.2635.pdf.
        self.B = torch.linalg.solve(torch.t(self.YPY), torch.t(self.A))
        self.t = torch.t(muX) - torch.matmul(torch.t(self.B), torch.t(muY))

    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the affine transformation.

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
        if Y is None:
            self.TY = torch.matmul(self.Y, self.B) + torch.tile(self.t, (self.M, 1))
            return
        else:
            return torch.matmul(Y, self.B) + torch.tile(self.t, (Y.shape[0], 1))

    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the affine transformation.
        See the update rule for sigma2 in Fig. 3 of of https://arxiv.org/pdf/0905.2635.pdf.

        """
        qprev = self.q

        trAB = torch.trace(torch.matmul(self.A, self.B))
        xPx = torch.matmul(torch.t(self.Pt1), torch.sum(
            torch.multiply(self.X_hat, self.X_hat), dim=1))
        trBYPYP = torch.trace(torch.matmul(torch.matmul(self.B, self.YPY), self.B))
        self.q = (xPx - 2 * trAB + trBYPYP) / (2 * self.sigma2) + \
            self.D * self.Np/2 * torch.log(self.sigma2)
        self.diff = torch.abs(self.q - qprev)

        self.sigma2 = (xPx - trAB) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

    def get_registration_parameters(self):
        """
        Return the current estimate of the affine transformation parameters.

        Returns
        -------
        B: numpy array
            DxD affine transformation matrix.

        t: numpy array
            1xD translation vector.

        """
        return self.B, self.t
