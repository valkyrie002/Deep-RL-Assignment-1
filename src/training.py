import copy
from time import perf_counter
from typing import Any, Literal, overload

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _as_tensor(values):
    if isinstance(values, torch.Tensor):
        return values.float()
    return torch.as_tensor(values, dtype=torch.float32)


@overload
def train_model(
    features: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    model: nn.Module,
    params: dict[str, Any],
    checkpoint: bool = False,
    ev_features: torch.Tensor | np.ndarray | None = None,
    ev_labels: torch.Tensor | np.ndarray | None = None,
    return_metrics: Literal[False] = False,
    rmse_scale: float = 1.0,
) -> nn.Module: ...


@overload
def train_model(
    features: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    model: nn.Module,
    params: dict[str, Any],
    checkpoint: bool = False,
    ev_features: torch.Tensor | np.ndarray | None = None,
    ev_labels: torch.Tensor | np.ndarray | None = None,
    return_metrics: Literal[True] = True,
    rmse_scale: float = 1.0,
) -> tuple[nn.Module, dict[str, Any]]: ...


def train_model(
    features,
    labels,
    model,
    params,
    checkpoint=False,
    ev_features=None,
    ev_labels=None,
    return_metrics=False,
    rmse_scale=1.0,
):
    learning_rate = params.get("learning_rate", 1e-3)
    learning_decay = float(params.get("learning_decay", 1.0))
    epochs = int(params.get("epoch", 10))
    batch_size = int(params.get("batch_size", len(features)))
    if epochs < 1:
        raise ValueError("epoch count must be at least 1")
    if batch_size < 1:
        raise ValueError("batch size must be at least 1")
    if learning_decay <= 0:
        raise ValueError("learning_decay must be positive")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    best_epoch = None
    epoch_eval_rmse = []

    features = _as_tensor(features)
    labels = _as_tensor(labels)
    if len(features) != len(labels):
        raise ValueError("features and labels must contain the same number of samples")
    if len(features) == 0:
        raise ValueError("training data must not be empty")
    has_evaluation_data = (
        ev_features is not None
        and ev_labels is not None
        and len(ev_features) > 0
        and len(ev_labels) > 0
    )
    if checkpoint and not has_evaluation_data:
        raise ValueError("checkpointing requires evaluation features and labels")
    if has_evaluation_data:
        ev_features = _as_tensor(ev_features)
        ev_labels = _as_tensor(ev_labels)
        if len(ev_features) != len(ev_labels):
            raise ValueError("evaluation features and labels must contain the same number of samples")

    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(int(params.get("seed", 0)))

    started_at = perf_counter()
    model.train()
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(features), generator=shuffle_generator)
        for batch_indices in permutation.split(batch_size):
            optimizer.zero_grad()
            output = model(features[batch_indices])
            loss = criterion(output, labels[batch_indices])
            loss.backward()
            optimizer.step()

        if has_evaluation_data:
            eval_rmse = evaluate_model_rmse(ev_features, ev_labels, model, params, scale=rmse_scale)
            epoch_eval_rmse.append({"epoch": epoch, "rmse": eval_rmse})
            if eval_rmse < best_val:
                best_val = eval_rmse
                best_epoch = epoch
                if checkpoint:
                    best_state = copy.deepcopy(model.state_dict())
            model.train()

        for param_group in optimizer.param_groups:
            param_group["lr"] *= learning_decay

    if checkpoint and best_state is not None:
        model.load_state_dict(best_state)
    model.train()
    training_time_seconds = perf_counter() - started_at

    if return_metrics:
        metrics = {
            "epoch_eval_rmse": epoch_eval_rmse,
            "best_eval_rmse": best_val if best_epoch is not None else None,
            "best_epoch": best_epoch,
            "training_time_seconds": training_time_seconds,
        }
        return model, metrics
    return model


def train_with_checkpoint(tr_features, tr_labels, ev_features, ev_labels, model, params):
    return train_model(
        tr_features,
        tr_labels,
        model,
        params,
        checkpoint=True,
        ev_features=ev_features,
        ev_labels=ev_labels,
    )


def tune_model(train_features, train_labels, eval_features, eval_labels, model, params):
    best_params = dict(params)
    best_model = model
    best_score = float("inf")

    learning_rates = [params.get("learning_rate", 1e-3) * scale for scale in (0.1, 1.0, 10.0)]
    for lr in learning_rates:
        candidate = copy.deepcopy(model)
        candidate_params = dict(params, learning_rate=lr)
        train_model(train_features, train_labels, candidate, candidate_params)
        score = evaluate_model(eval_features, eval_labels, candidate, candidate_params)

        if score.item() < best_score:
            best_score = score.item()
            best_model = candidate
            best_params = candidate_params

    return best_model, best_params


def evaluate_model(eval_features, eval_labels, model, params):
    criterion = nn.MSELoss()
    eval_features = _as_tensor(eval_features)
    eval_labels = _as_tensor(eval_labels)
    if len(eval_features) != len(eval_labels):
        raise ValueError("evaluation features and labels must contain the same number of samples")
    previous_mode = model.training
    model.eval()
    with torch.no_grad():
        output = model(eval_features)
        loss = criterion(output, eval_labels)
    model.train(previous_mode)
    return loss


def evaluate_model_rmse(eval_features, eval_labels, model, params, scale=1.0):
    return evaluate_model(eval_features, eval_labels, model, params).sqrt().item() * scale


def loss_function(predictions, labels):
    return nn.MSELoss()(predictions, labels)
