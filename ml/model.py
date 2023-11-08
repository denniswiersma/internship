#!/usr/bin/env python3

"""
An abstract base class that defines an interface for wrapper classes that
implement the usage of machine learning libraries. Subclasses provide, among
other things, an incapsulation of the complexities of the underlying library's
API, and a standardised interface for the analysis stage of this project.

Implementations are expected to override the abstract methods, and provide
additional case-specific methods as.
"""

# METADATA

# IMPORTS
import pickle as pkl
import secrets
from abc import ABC, abstractmethod
from pathlib import Path


# CLASSES
class Model(ABC):
    @abstractmethod
    def __init__(self):
        ...

    @staticmethod
    def _validate_path(path: Path):
        """
        Checks if a given path exists, and creates it if it does not.

        :param path: the path to a file. Please note this does not work with
        directories, since the parents of the path are created, not the path.
        """
        if not path.parent.exists():
            print(f"Path {path.parent} does not exist. Creating it now.")
            parent_dirs = path.parents
            for parent_dir in reversed(parent_dirs):
                if not parent_dir.exists():
                    parent_dir.mkdir()

    @staticmethod
    def _generate_runID() -> str:
        return secrets.token_hex(3)

    @abstractmethod
    def fit(self):
        self.fitted_model = self.fit()
        self.runID = self._generate_runID()

    @abstractmethod
    def predict(self, newx):
        ...

    @abstractmethod
    def assess(self, ytrue, ypred_proba):
        ...

    def save(self, path: Path):
        path = path.with_suffix(".pkl")
        self._validate_path(path)

        with open(path, "wb") as file:
            pkl.dump(self.fitted_model, file=file)
        print(f"Model saved to {path}")


# FUNCTIONS


def main():
    print("This file is meant to be imported, not run on it's own.")


if __name__ == "__main__":
    main()
