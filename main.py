import argparse
import itertools
import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.data_loader import (
    DATA_PATH,
    apply_standardization,
    fit_standardization,
    get_evaluation_data,
    get_test_data,
    get_training_subset,
)
from src.model import NeuralNetwork
from src.training import evaluate_model_rmse, train_model


DEFAULT_REPLICATE_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_LEARNING_RATES = [1e-4, 1e-3, 1e-2, 1e-1]
DEFAULT_TRAINING_SIZES = [1000, 10000, 100000]


def plot_rmse_vs_epoch(jsonl_path="out/run_data.jsonl", output_path="out/rmse_vs_epoch.png", show=False, step=20):
    """Plot mean evaluation RMSE by epoch for each training_size/learning_rate pair."""
    epoch_values = {}
    with open(jsonl_path, "r", encoding="utf-8") as run_file:
        for line in run_file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            training_size = record.get("training_size")
            learning_rate = record.get("learning_rate")
            if training_size is None or learning_rate is None:
                continue
            for point in record.get("epoch_eval_rmse", []):
                epoch = int(point.get("epoch"))
                if epoch % step != 0:
                    continue
                rmse = float(point.get("rmse"))
                key = (int(training_size), float(learning_rate))
                epoch_values.setdefault(key, {})
                epoch_values[key].setdefault(epoch, []).append(rmse)

    if not epoch_values:
        raise ValueError(f"No run data found in {jsonl_path!r}")

    fig, ax = plt.subplots(figsize=(11, 7))
    for (training_size, learning_rate), epoch_map in sorted(epoch_values.items(), key=lambda item: (item[0][0], item[0][1])):
        epochs = sorted(epoch_map)
        mean_rmse = [float(np.mean(epoch_map[epoch])) for epoch in epochs]
        label = f"train={training_size}, lr={learning_rate:g}"
        ax.plot(epochs, mean_rmse, marker="o", linewidth=2, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean evaluation RMSE")
    ax.set_title(f"RMSE vs. epoch (every {step} epochs), averaged across replicate seeds")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_rmse_by_configuration(jsonl_path="out/run_data.jsonl", out_dir="out/plots", step=20, show=False):
    """Create one RMSE-vs-epoch plot per training_size/learning_rate configuration."""
    configuration_data = {}
    with open(jsonl_path, "r", encoding="utf-8") as run_file:
        for line in run_file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            training_size = record.get("training_size")
            learning_rate = record.get("learning_rate")
            if training_size is None or learning_rate is None:
                continue
            key = (int(training_size), float(learning_rate))
            configuration_data.setdefault(key, {})
            for point in record.get("epoch_eval_rmse", []):
                epoch = int(point.get("epoch"))
                if epoch % step != 0:
                    continue
                configuration_data[key].setdefault(epoch, []).append(float(point.get("rmse")))

    if not configuration_data:
        raise ValueError(f"No run data found in {jsonl_path!r}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for (training_size, learning_rate), epoch_map in sorted(configuration_data.items(), key=lambda item: (item[0][0], item[0][1])):
        epochs = sorted(epoch_map)
        mean_rmse = [float(np.mean(epoch_map[epoch])) for epoch in epochs]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, mean_rmse, marker="o", linewidth=2, color="tab:blue")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean evaluation RMSE")
        ax.set_title(f"Training size={training_size}, learning rate={learning_rate:g}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        filename = f"train{training_size}_lr{learning_rate:g}.png"
        save_path = out_dir / filename
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        created.append(str(save_path))

    return created


def summarize_test_rmse_by_configuration(jsonl_path="out/run_data.jsonl"):
    """Summarize mean and standard deviation of test RMSE for each training_size/learning_rate configuration."""
    grouped = {}
    with open(jsonl_path, "r", encoding="utf-8") as run_file:
        for line in run_file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            training_size = record.get("training_size")
            learning_rate = record.get("learning_rate")
            test_rmse = record.get("test_rmse")
            if training_size is None or learning_rate is None or test_rmse is None:
                continue
            key = (int(training_size), float(learning_rate))
            grouped.setdefault(key, []).append(float(test_rmse))

    summary = {}
    for key, values in grouped.items():
        arr = np.asarray(values, dtype=float)
        summary[key] = {
            "training_size": key[0],
            "learning_rate": key[1],
            "mean_test_rmse": float(arr.mean()),
            "std_test_rmse": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "n_runs": len(arr),
            "test_rmse_values": values,
        }
    return summary


def initialize_experiment_model(hidden_layers=1, neurons_per_hidden_layer=64):
    return NeuralNetwork(
        in_dimension=14,
        out_dimension=1,
        hidden_layers=hidden_layers,
        neurons_per_hidden_layer=neurons_per_hidden_layer,
    )


def save_checkpoint_artifact(model, output_dir, run_record, normalization_stats):
    """Persist the best model state plus the normalization statistics needed to reuse it later."""
    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_record.get("run_id")
    if run_id is None:
        raise ValueError("run_record must include a run_id before saving a checkpoint")
    checkpoint_path = checkpoint_dir / f"{run_id}.pt"
    checkpoint_payload = {
        "run_id": run_id,
        "model_state_dict": model.state_dict(),
        "config": {
            "training_size": run_record.get("training_size"),
            "learning_rate": run_record.get("learning_rate"),
            "learning_decay": run_record.get("learning_decay"),
            "batch_size": run_record.get("batch_size"),
            "epochs": run_record.get("epochs"),
            "replicate_seed": run_record.get("replicate_seed"),
            "hidden_layers": run_record.get("hidden_layers"),
            "neurons_per_hidden_layer": run_record.get("neurons_per_hidden_layer"),
            "optimizer": run_record.get("optimizer"),
        },
        "normalization": {key: value.squeeze(0).tolist() if hasattr(value, "squeeze") else value for key, value in normalization_stats.items()},
        "metrics": {
            "best_eval_rmse": run_record.get("best_eval_rmse"),
            "best_epoch": run_record.get("best_epoch"),
            "training_time_seconds": run_record.get("training_time_seconds"),
        },
    }
    torch.save(checkpoint_payload, checkpoint_path)
    return checkpoint_path


def run_experiments(args):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"dataset not found: {DATA_PATH}")

    if args.sweep:
        training_sizes = [args.training_size] if args.training_size is not None else DEFAULT_TRAINING_SIZES
        learning_rates = [args.learning_rate] if args.learning_rate is not None else DEFAULT_LEARNING_RATES
    else:
        training_sizes = [args.training_size or 1000]
        learning_rates = [args.learning_rate or 1e-3]

    replicate_seeds = [args.replicate_seed] if args.replicate_seed is not None else DEFAULT_REPLICATE_SEEDS
    raw_evaluation_features, raw_evaluation_labels = get_evaluation_data()
    raw_test_features, raw_test_labels = get_test_data()
    training_data = {}
    normalization_data = {}
    for size in set(training_sizes):
        raw_training_features, raw_training_labels = get_training_subset(size)
        statistics = fit_standardization(raw_training_features, raw_training_labels)
        training_data[size] = apply_standardization(
            raw_training_features, raw_training_labels, statistics
        )
        evaluation_data = apply_standardization(
            raw_evaluation_features, raw_evaluation_labels, statistics
        )
        normalization_data[size] = (statistics, evaluation_data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as output_file:
        for training_size, learning_rate, replicate_seed in itertools.product(
            training_sizes,
            learning_rates,
            replicate_seeds,
        ):
            random.seed(replicate_seed)
            np.random.seed(replicate_seed)
            torch.manual_seed(replicate_seed)
            train_features, train_labels = training_data[training_size]
            statistics, (evaluation_features, evaluation_labels) = normalization_data[training_size]
            model = initialize_experiment_model(args.hidden_layers, args.neurons_per_hidden_layer)
            params = {
                "learning_rate": learning_rate,
                "learning_decay": args.learning_decay,
                "epoch": args.epochs,
                "batch_size": args.batch_size,
                "seed": replicate_seed,
            }
            run_started_at = datetime.now(timezone.utc).isoformat()
            run_id = uuid.uuid4().hex

            model, metrics = train_model(
                train_features,
                train_labels,
                model,
                params,
                checkpoint=args.checkpoint,
                ev_features=evaluation_features,
                ev_labels=evaluation_labels,
                return_metrics=True,
                rmse_scale=statistics["label_std"].item(),
            )
            test_features, test_labels = apply_standardization(raw_test_features, raw_test_labels, statistics)
            test_rmse = evaluate_model_rmse(
                test_features,
                test_labels,
                model,
                params,
                scale=statistics["label_std"].item(),
            )
            run_record = {
                "run_id": run_id,
                "run_started_at_utc": run_started_at,
                "dataset": str(DATA_PATH),
                "training_permutation_seed": 0,
                "replicate_seed": replicate_seed,
                "training_size": training_size,
                "learning_rate": learning_rate,
                "learning_decay": args.learning_decay,
                "hidden_layers": args.hidden_layers,
                "neurons_per_hidden_layer": args.neurons_per_hidden_layer,
                "input_dimension": 14,
                "output_dimension": 1,
                "batch_size": args.batch_size,
                "optimizer": "Adam",
                "epochs": args.epochs,
                "checkpoint_enabled": args.checkpoint,
                "test_rmse": test_rmse,
                "normalization": {
                    key: value.squeeze(0).tolist() for key, value in statistics.items()
                },
                **metrics,
            }
            if args.checkpoint:
                checkpoint_dir = output_path.parent / "checkpoints"
                checkpoint_path = save_checkpoint_artifact(model, checkpoint_dir, run_record, statistics)
                run_record["checkpoint_path"] = str(checkpoint_path)
            output_file.write(json.dumps(run_record) + "\n")
            output_file.flush()
            print(json.dumps(run_record))


def build_parser():
    parser = argparse.ArgumentParser(description="Train the assignment model on reproducible data subsets.")
    parser.add_argument("--train", action="store_true", help="run one configuration")
    parser.add_argument("--sweep", action="store_true", help="sweep learning rates and training sizes")
    parser.add_argument("--training-size", type=int, choices=DEFAULT_TRAINING_SIZES)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--learning-decay", type=float, default=1.0, help="multiply the learning rate by this factor after each epoch")
    parser.add_argument("--hidden-layers", type=int, default=1)
    parser.add_argument("--neurons-per-hidden-layer", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--replicate-seed", type=int, help="run one replicate; default runs seeds 0 through 4")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output", default="out/run_data.jsonl", help="append run records to this JSONL file")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.hidden_layers < 0 or args.neurons_per_hidden_layer < 1:
        parser.error("--hidden-layers must be non-negative and --neurons-per-hidden-layer must be positive")
    if args.learning_rate is not None and args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.learning_decay <= 0:
        parser.error("--learning-decay must be positive")
    if not args.train and not args.sweep:
        parser.error("select --train or --sweep")
    run_experiments(args)


if __name__ == "__main__":
    main()
