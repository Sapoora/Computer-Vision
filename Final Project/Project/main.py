"""Convenience CLI for training or evaluating the restoration project."""

from __future__ import annotations

import argparse

from training.evaluate import evaluate, parse_args as parse_eval_args
from training.train import parse_args as parse_train_args, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Computer Vision final project runner.")
    parser.add_argument("command", choices=["train", "evaluate"], help="Task to run.")
    args, remaining = parser.parse_known_args()

    if args.command == "train":
        train(parse_train_args(remaining))
    else:
        evaluate(parse_eval_args(remaining))


if __name__ == "__main__":
    main()
