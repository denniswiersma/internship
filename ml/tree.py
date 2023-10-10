#!/usr/bin/env python3

"""
"""

# METADATA

# IMPORTS
from pathlib import Path

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
        file_name = f"ctree_ts={teststat}_tt={testtype}_ss={splitstat}_st={splittest}_a={alpha}.png"
        print(file_name)
        # define control options
        control = ctree_control(
            teststat=teststat,
            testtype=testtype,
            splitstat=splitstat,
            splittest=splittest,
            alpha=alpha,
        )

        # build the tree
        model = ctree(
            formula=robjects.r.formula(model_formula),
            data=self.r_mm_with_tt,
            control=control,
        )

        # check if directory exists yet before trying to save to it

        if len(predictors) > 1:
            file_path = Path(
                f"ml/{predictors[0]}_{predictors[-1]}/" + file_name
            )
        else:
            file_path = Path(f"ml/{predictors[0]}/" + file_name)

        if not file_path.parent.exists():
            file_path.parent.mkdir()

        # save plot to disk
        grdevices = importr("grDevices")
        grdevices.png(
            file=file_path.as_posix(),
            width=5000,
            height=1500,
        )

        robjects.r.plot(
            model,
            margins=robjects.r.list(15, 0, 0, 0),
            tp_args=robjects.r.list(rot=90, just=robjects.r.c("right", "top")),
        )

        grdevices.dev_off()

        return file_path


# FUNCTIONS


def main():
    print("This file is meant to be imported, not run on it's own.")


if __name__ == "__main__":
    main()
