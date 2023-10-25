#!/usr/bin/env python3

"""
"""

# METADATA

import pickle as pkl
from pathlib import Path
# IMPORTS
from typing import Callable

import numpy as np
import pandas as pd
from rpy2 import robjects
from rpy2.robjects.packages import importr

from data import Data


# CLASSES
class TreeBuilder:
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

    def ctree(
        self,
        teststat: str = "quad",
        testtype: str = "Bonferroni",
        splitstat: str = "quad",
        splittest: bool = False,
        alpha: float = 0.05,
        predictors: list = ["consensus independent component 1"],
        response: str = "response",
    ):
        return

    def save_tree(
        self,
        model,
        teststat: str = "quad",
        testtype: str = "Bonferroni",
        splitstat: str = "quad",
        splittest: bool = False,
        alpha: float = 0.05,
        predictors: list = ["consensus independent component 1"],
        type: str | list[str] = ["img", "model"],
    ):
        # construct a string describing the tree's settings
        file_name = f"ctree_ts={teststat}_tt={testtype}_ss={splitstat}_st={splittest}_a={alpha}"
        print(file_name)

        # construct path to file based on predictors
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
            print("saved model")

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

    def build_ctree(
        self,
        teststat: str = "quad",
        testtype: str = "Bonferroni",
        splitstat: str = "quad",
        splittest: bool = False,
        alpha: float = 0.05,
        predictors: list = ["consensus independent component 1"],
        response: str = "response",
    ):
        if splittest and not isinstance(testtype, np.ndarray):
            if testtype != "MonteCarlo":
                return

        model_formula = self.build_formula(predictors.copy(), response)

        # import partykit and make objects
        importr("partykit")
        ctree = robjects.r["ctree"]
        ctree_control = robjects.r["ctree_control"]

        print("building tree...")
        # define control options
        control = ctree_control(  # type: ignore
            teststat=teststat,
            testtype=testtype,
            splitstat=splitstat,
            splittest=splittest,
            alpha=alpha,
        )

        # build the tree
        model = ctree(  # type: ignore
            formula=robjects.r.formula(model_formula),  # type: ignore
            data=self.r_mm_with_tt,
            control=control,
        )

        return model


# FUNCTIONS


def main():
    print("This file is meant to be imported, not run on it's own.")


if __name__ == "__main__":
    main()
