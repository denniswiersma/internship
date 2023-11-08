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
    def generate_runID() -> str:
        return secrets.token_hex(3)

    @abstractmethod
    def fit(self):
        self.fitted_model = self.fit()

    @abstractmethod
    def predict(self, newx):
        ...

    @abstractmethod
    def assess(self, ytrue, ypred_proba):
        ...

    def save(self, path: Path):
        path = path.with_suffix(".pkl")

        with open(path, "wb") as file:
            pkl.dump(self.fitted_model, file=file)
        print(f"Model saved to {path}")


# FUNCTIONS


def main():
    print("This file is meant to be imported, not run on it's own.")


if __name__ == "__main__":
    main()
