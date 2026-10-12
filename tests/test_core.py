import torch

from src.model import NeuralNetwork
from src.data_loader import (
    apply_standardization,
    denormalize_labels,
    fit_standardization,
    get_training_subset,
)
from src.training import train_model, evaluate_model, evaluate_model_rmse


def test_neural_network_forward_pass():
    model = NeuralNetwork(3, 1, 2, 4)
    features = torch.randn(8, 3)
    output = model(features)

    assert output.shape == (8, 1)
    assert torch.isfinite(output).all()


def test_train_model_reduces_loss():
    model = NeuralNetwork(3, 1, 1, 8)
    features = torch.randn(32, 3)
    labels = torch.randn(32, 1)
    params = {"learning_rate": 1e-2, "epoch": 25}

    initial_loss = evaluate_model(features, labels, model, params)
    train_model(features, labels, model, params)
    final_loss = evaluate_model(features, labels, model, params)

    assert final_loss < initial_loss


def test_train_model_applies_learning_decay_each_epoch(monkeypatch):
    model = NeuralNetwork(3, 1, 1, 8)
    features = torch.randn(32, 3)
    labels = torch.randn(32, 1)

    observed_lrs = []

    class RecordingAdam:
        def __init__(self, params, lr):
            self.param_groups = [{"params": list(params), "lr": lr}]
            observed_lrs.append(float(self.param_groups[0]["lr"]))

        def zero_grad(self):
            pass

        def step(self):
            observed_lrs.append(float(self.param_groups[0]["lr"]))

    monkeypatch.setattr("src.training.optim.Adam", RecordingAdam)

    train_model(
        features,
        labels,
        model,
        {"learning_rate": 1e-2, "learning_decay": 0.5, "epoch": 3, "batch_size": 8, "seed": 0},
    )

    assert observed_lrs[0] == 1e-2
    assert observed_lrs[-1] == 1e-2 * (0.5 ** 2)


def test_checkpoint_uses_validation_improvement_only():
    model = NeuralNetwork(3, 1, 1, 8)
    train_features = torch.randn(64, 3)
    train_labels = torch.randn(64, 1)
    eval_features = torch.randn(16, 3)
    eval_labels = torch.randn(16, 1)
    params = {"learning_rate": 1e-2, "epoch": 30}

    initial_eval_loss = evaluate_model(eval_features, eval_labels, model, params)
    trained = train_model(
        train_features,
        train_labels,
        model,
        params,
        checkpoint=True,
        ev_features=eval_features,
        ev_labels=eval_labels,
    )
    final_eval_loss = evaluate_model(eval_features, eval_labels, trained, params)

    assert trained.training is True
    assert final_eval_loss <= initial_eval_loss


def test_training_subsets_are_fixed_nested_prefixes():
    small_features, small_labels = get_training_subset(1000)
    medium_features, medium_labels = get_training_subset(10000)

    assert torch.equal(small_features, medium_features[:1000])
    assert torch.equal(small_labels, medium_labels[:1000])


def test_training_returns_epoch_rmse_and_best_epoch():
    model = NeuralNetwork(3, 1, 1, 8)
    train_features = torch.randn(32, 3)
    train_labels = torch.randn(32, 1)
    eval_features = torch.randn(12, 3)
    eval_labels = torch.randn(12, 1)
    params = {"learning_rate": 1e-2, "epoch": 4, "batch_size": 8, "seed": 3}

    _, metrics = train_model(
        train_features,
        train_labels,
        model,
        params,
        ev_features=eval_features,
        ev_labels=eval_labels,
        return_metrics=True,
        rmse_scale=3.5,
    )

    assert len(metrics["epoch_eval_rmse"]) == 4
    assert metrics["best_epoch"] in range(1, 5)
    assert metrics["best_eval_rmse"] == min(item["rmse"] for item in metrics["epoch_eval_rmse"])
    assert metrics["training_time_seconds"] >= 0
    expected_raw_rmse = evaluate_model(eval_features, eval_labels, model, params).sqrt().item() * 3.5
    assert abs(metrics["epoch_eval_rmse"][-1]["rmse"] - expected_raw_rmse) < 1e-6


def test_evaluate_model_rmse_matches_expected_value():
    model = NeuralNetwork(3, 1, 1, 8)
    features = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
    labels = torch.tensor([[1.0], [2.0], [3.0]])
    scale = 2.5

    raw_rmse = evaluate_model_rmse(features, labels, model, {}, scale=scale)
    expected = evaluate_model(features, labels, model, {}).sqrt().item() * scale

    assert abs(raw_rmse - expected) < 1e-6


def test_standardization_is_fitted_from_training_data_and_reversible():
    train_features = torch.tensor([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
    train_labels = torch.tensor([[2.0], [6.0], [10.0]])
    eval_features = torch.tensor([[101.0, 210.0]])
    eval_labels = torch.tensor([[202.0]])

    statistics = fit_standardization(train_features, train_labels)
    normalized_features, normalized_labels = apply_standardization(
        train_features, train_labels, statistics
    )
    normalized_eval_features, normalized_eval_labels = apply_standardization(
        eval_features, eval_labels, statistics
    )

    assert torch.allclose(normalized_features.mean(dim=0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(normalized_features.std(dim=0, correction=0), torch.ones(2), atol=1e-6)
    assert torch.allclose(normalized_labels.mean(dim=0), torch.zeros(1), atol=1e-6)
    assert torch.allclose(denormalize_labels(normalized_labels, statistics), train_labels)
    assert normalized_eval_features.abs().max() > 1
    assert normalized_eval_labels.abs().max() > 1
