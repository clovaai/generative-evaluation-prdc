import torch
from typing import Dict


def nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: torch.Tensor([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = torch.cdist(input_features, input_features)
    radii = torch.kthvalue(distances, k=nearest_k + 1, dim=-1)[0]
    return radii


def compute_prdc_torch(real_features, fake_features, nearest_k):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        real_features: torch.Tensor([N, feature_dim], dtype=torch.float32)
        fake_features: torch.Tensor([N, feature_dim], dtype=torch.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """

    print('Num real: {} Num fake: {}'
          .format(real_features.shape[0], fake_features.shape[0]))

    real_nearest_neighbour_distances = nearest_neighbour_distances(
        real_features, nearest_k)
    fake_nearest_neighbour_distances = nearest_neighbour_distances(
        fake_features, nearest_k)
    distance_real_fake = torch.cdist(
            real_features, fake_features)

    precision = (
            distance_real_fake < real_nearest_neighbour_distances[:, None]
    ).any(dim=0).double().mean().item()

    recall = (
            distance_real_fake < fake_nearest_neighbour_distances[None, :]
    ).any(dim=1).double().mean().item()

    density = (1. / float(nearest_k)) * (
            distance_real_fake < real_nearest_neighbour_distances[:, None]
    ).sum(dim=0).double().mean().item()

    coverage = (
            distance_real_fake.min(dim=1)[0] <
            real_nearest_neighbour_distances
    ).double().mean().item()

    return dict(precision=precision, recall=recall,
                density=density, coverage=coverage)
