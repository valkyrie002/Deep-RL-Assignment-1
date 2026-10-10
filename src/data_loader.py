from pathlib import Path

import numpy as np
import torch


DATA_PATH = Path(__file__).with_name("swept_volume_data.npz")
TRAINING_PERMUTATION_SEED = 0


def load_dataset(data_path=DATA_PATH, keys=None):
    with np.load(data_path) as archive:
        selected_keys = archive.files if keys is None else keys
        return {key: torch.as_tensor(archive[key].copy(), dtype=torch.float32) for key in selected_keys}


def fixed_training_permutation(num_samples, seed=TRAINING_PERMUTATION_SEED):
    return np.random.default_rng(seed).permutation(num_samples)


def get_training_subset(data_size, data_path=DATA_PATH):
    dataset = load_dataset(data_path, ("training_features", "training_labels"))
    features = dataset["training_features"]
    labels = dataset["training_labels"]
    if data_size < 1 or data_size > len(features):
        raise ValueError(f"training size must be between 1 and {len(features)}")

    indices = fixed_training_permutation(len(features))[:data_size]
    indices = torch.as_tensor(indices, dtype=torch.long)
    return features[indices], labels[indices]


def get_evaluation_data(data_path=DATA_PATH):
    dataset = load_dataset(data_path, ("evaluation_features", "evaluation_labels"))
    return dataset["evaluation_features"], dataset["evaluation_labels"]


def get_test_data(data_path=DATA_PATH):
    dataset = load_dataset(data_path, ("testing_features", "testing_labels"))
    return dataset["testing_features"], dataset["testing_labels"]


def fit_standardization(features, labels):
    feature_mean = features.mean(dim=0, keepdim=True)
    feature_std = features.std(dim=0, keepdim=True, correction=0)
    label_mean = labels.mean(dim=0, keepdim=True)
    label_std = labels.std(dim=0, keepdim=True, correction=0)

    feature_std = torch.where(feature_std > 0, feature_std, torch.ones_like(feature_std))
    label_std = torch.where(label_std > 0, label_std, torch.ones_like(label_std))
    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "label_mean": label_mean,
        "label_std": label_std,
    }


def apply_standardization(features, labels, statistics):
    normalized_features = (features - statistics["feature_mean"]) / statistics["feature_std"]
    normalized_labels = (labels - statistics["label_mean"]) / statistics["label_std"]
    return normalized_features, normalized_labels


def denormalize_labels(labels, statistics):
    return labels * statistics["label_std"] + statistics["label_mean"]
