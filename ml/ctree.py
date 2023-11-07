#!/usr/bin/env python3

"""
"""

# METADATA

# IMPORTS
import dataclasses
import math
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rpy2 import robjects
from rpy2.robjects.packages import importr

from data import Data


# CLASSES
class Ctree:
    def __init__(self, data: Data):
        self.mixing_matrix: pd.DataFrame = data.mixing_matrix
        self.tumor_types: pd.DataFrame = data.tumor_types
        self.mm_with_tt: pd.DataFrame = data.get_mm_with_tt()
        self.r_mm_with_tt = data.get_r_mm_with_tt()

    def build_formula(
        self,
        predictors: list = ["consensus independent component 1"],
        response: str = "response",
    ):
        # build formula to be passed to the model
        model_formula: str = f"{response} ~ "

        # handle first case since it should not include the "+" character
        first_predictor: str = predictors.pop(0)
        model_formula += f"`{first_predictor}`"

        # handle remaining predictors
        for predictor in predictors:
            model_formula += f" + `{predictor}`"

        return model_formula

    @dataclass
    class ctree_control:
        testtype: str = "Bonferroni"  # how to compute the distribution of the test statistic
        alpha: float = 0.05  # significance level for variable selection
        maxdepth: float | int = math.inf  # maximum depth of the tree
        minsplit: int = 20  # minimum sum of weights in a node to be considered for splitting
        minbucket: int = 7  # minimum sum of weights in a terminal node

    def ctree(
        self,
        ctree_control: ctree_control,
        predictors: list = ["consensus independent component 1"],
        # TODO: because of train_test_split, a given ncolumns will only be included in the dataset
        # therefore, the formula should be constructed based on all the columns in the passed dataset
        response: str = "response",
    ):
        # TODO: add pydoc
        model_formula = self.build_formula(predictors.copy(), response)

        # import partykit and make objects
        importr("partykit")
        ctree = robjects.r["ctree"]
        ctree_control_func = robjects.r["ctree_control"]

        print("building tree...")  # TODO: use logging
        # define control options
        control = ctree_control_func(**dataclasses.asdict(ctree_control))  # type: ignore

        # build the tree
        model = ctree(  # type: ignore
            formula=robjects.r.formula(model_formula),  # type: ignore
            # TODO: use data argument (since you need to use training data)
            data=self.r_mm_with_tt,
            control=control,
        )

        return model

    # TODO: implement plot function that returns figure and axes
    # and let the user save the figure themselves from the ipynb.
    # save_tree shall only save the serialised model to disk.

    def save_tree(
        self,
        model,
        ctree_control: ctree_control,
        predictors: list = ["consensus independent component 1"],
        type: str | list[str] = ["img", "model"],
    ):
        # TODO: add pydoc

        # construct a string describing the tree's settings
        file_name = "ctree"
        for key, value in dataclasses.asdict(ctree_control).items():
            file_name += f"-{key}={str(value).replace('.', '_')}"

        # construct path to file based on predictors
        # TODO: don't do these folders anymore, perhaps do generate a run ID.
        if len(predictors) > 1:
            file_path = Path(
                f"ml/{predictors[0]}_{predictors[-1]}/" + file_name
            )
        else:
            file_path = Path(f"ml/{predictors[0]}/" + file_name)

        if not file_path.parent.exists():
            file_path.parent.mkdir()

        def save_image(file_path):
            # append file extension
            file_path = file_path.with_suffix(".png")
            # import the graphics device in order to enable saving images to
            # disk
            grdevices = importr("grDevices")
            grdevices.png(
                file=file_path.as_posix(),
                width=5000,
                height=1500,
            )

            # plot the tree and save to disk
            robjects.r.plot(  # type: ignore
                model,
                margins=robjects.r.list(15, 0, 0, 0),  # type: ignore
                tp_args=robjects.r.list(  # type: ignore
                    rot=90, just=robjects.r.c("right", "top")  # type: ignore
                ),
            )
            # disable graphics device
            grdevices.dev_off()
            print(f"saved image at {file_path}")

        def save_model(file_path):
            # append file extension
            file_path = file_path.with_suffix(".pkl")

            with open(file_path, "wb") as file:
                pkl.dump(model, file=file)
            print(f"saved model at {file_path}")

        match type:
            case "img":
                save_image(file_path)
            case "model":
                save_model(file_path)
            case ["img", "model"] | ["model", "img"]:
                save_image(file_path)
                save_model(file_path)
            case _:
                raise ValueError(
                    f"{type} not of valid values 'img' 'model' ['img', 'model']"
                )


# FUNCTIONS


def main():
    print("This file is meant to be imported, not run on it's own.")


if __name__ == "__main__":
    main()
