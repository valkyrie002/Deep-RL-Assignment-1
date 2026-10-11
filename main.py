import argparse
import itertools
import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.data_loader import (
    DATA_PATH,
    apply_standardization,
    fit_standardization,
    get_evaluation_data,
    get_training_subset,
)
from src.model import NeuralNetwork
from src.training import train_model


DEFAULT_REPLICATE_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_LEARNING_RATES = [1e-4, 1e-3, 1e-2, 1e-1]
DEFAULT_TRAINING_SIZES = [1000, 10000, 100000]


def initialize_experiment_model(hidden_layers=1, neurons_per_hidden_layer=64):
    return NeuralNetwork(
        in_dimension=14,
        out_dimension=1,
        hidden_layers=hidden_layers,
        neurons_per_hidden_layer=neurons_per_hidden_layer,
    )


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
                "epoch": args.epochs,
                "batch_size": args.batch_size,
                "seed": replicate_seed,
            }
            run_started_at = datetime.now(timezone.utc).isoformat()

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
            run_record = {
                "run_id": uuid.uuid4().hex,
                "run_started_at_utc": run_started_at,
                "dataset": str(DATA_PATH),
                "training_permutation_seed": 0,
                "replicate_seed": replicate_seed,
                "training_size": training_size,
                "learning_rate": learning_rate,
                "hidden_layers": args.hidden_layers,
                "neurons_per_hidden_layer": args.neurons_per_hidden_layer,
                "input_dimension": 14,
                "output_dimension": 1,
                "batch_size": args.batch_size,
                "optimizer": "Adam",
                "epochs": args.epochs,
                "checkpoint_enabled": args.checkpoint,
                "normalization": {
                    key: value.squeeze(0).tolist() for key, value in statistics.items()
                },
                **metrics,
            }
            output_file.write(json.dumps(run_record) + "\n")
            output_file.flush()
            print(json.dumps(run_record))


def build_parser():
    parser = argparse.ArgumentParser(description="Train the assignment model on reproducible data subsets.")
    parser.add_argument("--train", action="store_true", help="run one configuration")
    parser.add_argument("--sweep", action="store_true", help="sweep learning rates and training sizes")
    parser.add_argument("--training-size", type=int, choices=DEFAULT_TRAINING_SIZES)
    parser.add_argument("--learning-rate", type=float)
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
    if not args.train and not args.sweep:
        parser.error("select --train or --sweep")
    run_experiments(args)


if __name__ == "__main__":
    main()
