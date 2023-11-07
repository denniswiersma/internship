#!/usr/bin/env python3

"""
An abstract base class that defines an interface for wrapper classes that
implement the usage of machine learning libraries. Subclasses provide, among
other things, an incapsulation of the complexities of the underlying library's
API, and a standardised interface for the analysis stage of this project.

Implementations are expected to override the abstract methods, and provide
additional case-specific methods as required.
"""

# METADATA

# IMPORTS
from abc import ABC, abstractmethod


# CLASSES
class Model(ABC):
    @abstractmethod
    def __init__(self):
        ...

    @abstractmethod
    def fit(self):
        ...

    @abstractmethod
    def predict(self, model, newx):
        ...

    @abstractmethod
    def assess(self, ytrue, ypred_proba):
        ...

    @abstractmethod
    def save(self):
        ...


# FUNCTIONS


def main():
    print("This file is meant to be imported, not run on it's own.")


if __name__ == "__main__":
    main()
