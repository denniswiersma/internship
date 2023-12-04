#!/usr/bin/env python3

"""
"""

# METADATA

# IMPORTS
import argparse
import re
from pathlib import Path

import pandas as pd


# CLASSES
class LogParser:
    def __init__(self):
        self._args = self._parse_args()

        self.input = Path(self._args.input)
        self.output = Path(self._args.output)

        self.nsets = None

    def _parse_args(self):
        parser = argparse.ArgumentParser(
            prog="log_parser.py",
            description="Parse log files to extract settings and performance metrics.",
        )

        parser.add_argument(
            "-i",
            "--input",
            type=str,
            required=True,
            help="Path to the directory containing analysis output.",
        )

        parser.add_argument(
            "-o",
            "--output",
            type=str,
            required=False,
            help="Save parsed output to file.",
        )

        return parser.parse_args()

    def find_log_files(self):
        paths = []

        for dirpath, _, files in self.input.walk():
            for file in files:
                file = Path(file)
                if file.suffix == ".log":
                    paths.append(dirpath / file)

        print(f"Found {len(paths)} log files")
        self.paths = paths

    def parse_log_file(self, path: Path):
        performance_buffer = []
        file_settings = {}
        time = 0
        ntrees = 0

        with open(path, "r") as f:
            for line in f:
                split_line = line.split(":")[-1].strip()

                if re.match(r"^\w+\.CtreeControl\(|\)$", split_line):
                    file_settings = self.extract_settings(split_line)

                if re.match(r"^[\w\s-]+=\s+[0-9.]+", line) and line.split("=")[
                    0
                ].strip() not in ["train", "test", "val"]:
                    performance_buffer.append(line)

                if re.match(r"\w+ built in \d+\.?\d+ minutes", split_line):
                    time = self.extract_time(split_line)

                if re.match(r"number of trees = \d+", split_line):
                    ntrees = self.extract_ntrees(split_line)

        performance = self.extract_performance(performance_buffer)

        return file_settings, performance, time, ntrees

    def extract_settings(self, line):
        settings_str = re.sub(r"^\w+\.CtreeControl\(|\)$", "", line)
        pairs = settings_str.split(", ")

        settings = {}
        for pair in pairs:
            key, value = pair.split("=")

            if value == "inf":
                value = float("inf")
            elif re.match(r"^[0-9]+\.?[0-9]*$", value):
                value = float(value)
            elif value.startswith("'") and value.endswith("'"):
                value = value.strip("'")

            settings[key] = value
        return settings

    def extract_performance(
        self, performance_buffer: list
    ) -> dict[str, dict[str, float]]:
        performances = {
            "metrics": [],
            "values": [],
        }
        for line in performance_buffer:
            line = line.split("=")
            metric = line[0].strip()
            value = float(line[1].strip())
            performances["metrics"].append(metric)
            performances["values"].append(value)

        unique_metrics = set(performances["metrics"])

        if not self.nsets:
            nsets = int(len(performances["metrics"]) / len(unique_metrics))

            if nsets == 1:
                print(
                    "Found one set of performance metrics, assuming validation data"
                )
                self.sets = ["val"]
            elif nsets == 2:
                print(
                    "Found two sets of performance metrics, assuming training and validation data"
                )
                self.sets = ["train", "val"]
            elif nsets == 3:
                print(
                    "Found three sets of performance metrics, assuming training, validation, and testing data"
                )
                print(
                    "Please note one should not use testing data for hyperparameter tuning"
                )
                self.sets = ["train", "val", "test"]
            else:
                raise ValueError(f"Expected no more than 3 sets, got {nsets}")
            self.nsets = nsets

        perfs = {mset: {} for mset in self.sets}

        for i, (metric, value) in enumerate(
            zip(performances["metrics"], performances["values"])
        ):
            set_index = i // len(unique_metrics)
            mset = self.sets[set_index]
            if metric not in perfs[mset]:
                perfs[mset][metric] = value

        return perfs

    def extract_time(self, line):
        return line.split(" ")[-2]

    def extract_ntrees(self, line):
        return int(line.split(" ")[-1])


def create_dataframe(parser):
    rows = []

    for path in parser.paths:
        settings, performance, time, ntrees = parser.parse_log_file(path)

        flattened_performance = {
            f"{mset}_{metric}": value
            for mset, metrics in performance.items()
            for metric, value in metrics.items()
        }

        record = {
            "runtime": time,
            "ntrees": ntrees,
            **settings,
            **flattened_performance,
        }
        rows.append(record)

    return pd.DataFrame(rows)


# FUNCTIONS


def main():
    parser = LogParser()
    parser.find_log_files()
    df = create_dataframe(parser)
    print(df)


if __name__ == "__main__":
    main()
