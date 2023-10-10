#!/usr/bin/env python3

"""
"""

# METADATA

# IMPORTS
from data import Data
import itertools
from pathlib import Path

import pandas as pd
import numpy as np
from datetime import datetime

from rpy2.robjects.packages import importr
from rpy2 import robjects
import multiprocessing as mp


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
        response: str = "TYPE3",
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
        response: str = "TYPE3",
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
                f"tree/{predictors[0]}_{predictors[-1]}/" + file_name
            )
        else:
            file_path = Path(f"tree/{predictors[0]}/" + file_name)

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


# FUNCTIONS


def main():
    # print starting timestamp
    now = datetime.now()
    print(f"starting time: {now.time()}")

    # load data and pass to treebuilder
    data = Data(data_folder="data/")
    treebuilder = TreeBuilder(data)

    # define parameters to be passed to ctree
    teststats: list[str] = ["quad", "max"]
    testtypes: list[str | list] = [
        "Teststatistic",
        "Univariate",
        "Bonferroni",
        "MonteCarlo",
        robjects.r.c("MonteCarlo", "Bonferroni"),
    ]
    splitstats: list[str] = ["quad", "max"]
    # splittests: list[bool] = [True, False]
    alphas: list[float] = [0.1, 0.05, 0.01]
    predictors: list[list[str]] = [
        ["consensus independent component 1"],
        ["consensus independent component 2"],
        ["consensus independent component 3"],
    ]

    # make all possible combinations of parameters above
    arg_combos = itertools.product(
        teststats, testtypes, splitstats, alphas, predictors
    )

    # i = 0
    # for x in arg_combos:
    #     i += 1
    # print(i)

    t2 = datetime.now()
    # with mp.Pool(1) as p:
    #     p.starmap(treebuilder.build_ctree, arg_combos)

    treebuilder.build_ctree(
        testtype=robjects.r.c("MonteCarlo", "Bonferroni"),
        teststat="quad",
        splittest=False,
        predictors=[
            "consensus independent component 1",
        ],
    )
    print(datetime.now() - t2)


if __name__ == "__main__":
    main()
