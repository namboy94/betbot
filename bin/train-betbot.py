#!/usr/bin/env python3
"""LICENSE
Copyright 2017 Hermann Krumrey <hermann@krumreyh.com>

This file is part of bundesliga-tippspiel.

bundesliga-tippspiel is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

bundesliga-tippspiel is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with bundesliga-tippspiel.  If not, see <http://www.gnu.org/licenses/>.
LICENSE"""

import os
import argparse
from betbot import sentry_dsn
from keras.models import Sequential
from keras.layers import Dense, Flatten
from puffotter.init import cli_start, argparse_add_verbosity
from betbot.neural.keras.BetOddsTrainer import BetOddsTrainer
from betbot.neural.keras.TableHistoryTrainer import TableHistoryTrainer


def main(args: argparse.Namespace):
    """
    Trains the betbot neural networks and evaluates them
    :param args: The command line arguments
    :return: None
    """
    if args.trainer == "betodds":
        trainer = BetOddsTrainer(args.output_dir)
    elif args.trainer == "tablehistory":
        trainer = TableHistoryTrainer(args.output_dir)
    else:
        return

    trainer.name = args.name

    if args.refresh_training_data:
        trainer.load_training_data(True)

    def gen_model(h, o):
        return Sequential([
            Flatten(input_shape=(12,)),
            Dense(20, activation=h),
            Dense(2, activation=o)
        ])

    custom_models = [
        (lambda: gen_model("sigmoid", "relu"), "sigmoid-relu"),
        (lambda: gen_model("sigmoid", "relu"), "sigmoid-linear"),
        (lambda: gen_model("sigmoid", "exponential"), "sigmoid-exponential"),
        (lambda: gen_model("linear", "linear"), "linear-linear")
    ]
    custom_compilations = [
        (lambda m: m.compile(loss="mae", optimizer="sgd"), "mae-sgd"),
        (lambda m: m.compile(loss="mae", optimizer="adamax"), "mae-adamax"),
        (lambda m: m.compile(loss="mae", optimizer="nadam"), "mae-nadam")
    ]
    if not args.try_parameters:
        custom_models = [(None, "default")]
        custom_compilations = custom_models

    for custom_model in custom_models:
        for custom_compilation in custom_compilations:
            name = f"{args.name}-{custom_model[1]}-{custom_compilation[1]}"
            trainer.name = name
            model, score, accuracy, avg_score, avg_accuracy = trainer.train(
                args.iterations,
                args.epochs,
                args.batch_size,
                custom_model_fn=custom_model[0],
                custom_compile_fn=custom_compilation[0]
            )
            print(f"Best Score: {score}, "
                  f"Best Accuracy: {accuracy:.2f}%")
            print(f"Average Score: {avg_score}, "
                  f"Average Accuracy: {avg_accuracy:.2f}%")
            name = f"[{avg_score:.2f}][{score:.2f}] {name}"
            model.save(os.path.join(args.output_dir, name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("name")
    parser.add_argument("trainer", choices={"betodds", "tablehistory"})
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--refresh-training-data", action="store_true")
    parser.add_argument("--try-parameters", action="store_true")
    argparse_add_verbosity(parser)
    cli_start(main, parser, "Thanks for using betbot", "betbot", sentry_dsn)
