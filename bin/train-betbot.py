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

import argparse
from betbot import sentry_dsn
from puffotter.init import cli_start, argparse_add_verbosity
from betbot.neural.keras.TableHistoryTrainer import TableHistoryTrainer


def main(args: argparse.Namespace):
    """
    Trains the betbot neural networks and evaluates them
    :param args: The command line arguments
    :return: None
    """
    trainer = TableHistoryTrainer(args.output_dir)
    trainer.name = args.name
    model, score = trainer.train(args.iterations, args.epochs, args.batch_size)
    print(f"Score: {score}")
    #model.save(trainer.model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("name")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=25)
    argparse_add_verbosity(parser)
    cli_start(main, parser, "Thanks for using betbot", "betbot", sentry_dsn)
