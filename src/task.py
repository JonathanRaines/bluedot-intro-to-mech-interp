"""
This module defines the task by generating every permutation of the numbers 0 to p-1 and the equals sign, and then calculating the sum of the first two numbers modulo p.

The data is split into training and testing sets, and the training data and labels are returned as tensors.
"""

import einops
import torch


def make_dataset_and_labels(p: int, device: str):
    # The input to the transformer is a sequence of three tokens |a|b|=|
    a_vector: torch.Tensor = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
    b_vector: torch.Tensor = einops.repeat(torch.arange(p), "j -> (i j)", i=p)

    # We'll use 0 to p-1 as numbers, and p as the token for the = sign.
    equals_vector: torch.Tensor = einops.repeat(torch.tensor(p), " -> (i j)", i=p, j=p)

    dataset: torch.tensor = torch.stack([a_vector, b_vector, equals_vector], dim=1).to(
        device
    )
    labels: torch.tensor = (dataset[:, 0] + dataset[:, 1]) % p

    return dataset, labels


def make_train_and_test_data(
    p: int, device: str, data_seed: int, training_fraction: float
) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """
    Generate data and labels for training and testing a transformer model.

    Args:
        p (int): The base for modular arithmetic and number of tokens in the vocabulary.
        device (str): The device to store the tensors on.
        data_seed (int): The seed value for random data generation.
        training_fraction (float): The fraction of data to use for training.

    Returns:
        tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]: A tuple containing the training data, training labels, testing data, and testing labels.
    """
    dataset, labels = make_dataset_and_labels(p, device)

    torch.manual_seed(data_seed)

    indices: torch.Tensor = torch.randperm(p**2)
    cutoff: int = int(p**2 * training_fraction)
    train_indices = indices[:cutoff]
    test_indices = indices[cutoff:]

    # These will all also be on device because dataset and labels are on device.
    train_data: torch.Tensor = dataset[train_indices]
    train_labels: torch.Tensor = labels[train_indices]
    test_data: torch.Tensor = dataset[test_indices]
    test_labels: torch.Tensor = labels[test_indices]

    return train_data, train_labels, test_data, test_labels
