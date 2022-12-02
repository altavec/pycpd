import torch


def is_positive_semi_definite(R):
    if not isinstance(R, torch.Tensor):
        raise ValueError('Encountered an error while checking if the matrix is positive semi definite. \
            Expected a numpy array, instead got : {}'.format(R))
    return torch.all(torch.linalg.eigvals(R) > 0)
