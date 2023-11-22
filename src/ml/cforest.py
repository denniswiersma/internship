#!/usr/bin/env python3

"""
"""

# METADATA

# IMPORTS
import dataclasses
import math
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from src.data import Data
from src.log_manager import LogManager
from src.ml.model import Model


# CLASSES
class Cforest(Model):
    def __init__(self, data: Data):
        self.mm_with_tt: pd.DataFrame = data.get_mm_with_tt()
        self.data: Data = data
        self.lm = LogManager()

    @dataclass
    class CtreeControl:
        testtype: str = "Univariate"  # how to compute the distribution of the test statistic
        alpha: float = 0.05  # significance level for variable selection
        maxdepth: float | int = math.inf  # maximum depth of the tree
        minsplit: int = 20  # minimum sum of weights in a node to be considered for splitting
        minbucket: int = 7  # minimum sum of weights in a terminal node

    def fit(
        self,
        train,
        ctree_control: CtreeControl,
        ntree: int = 500,
        cores: int = 1,
    ):
        self.runID = self._generate_runID()

        output_dir = Path(
            self.data.config["output"]["locations"]["cforest"]
        ).joinpath(self.runID, "cforest")
        self.lm.add_file_handler(path=output_dir)

        if any([self.lm.load_buffer, self.lm.prep_buffer, self.lm.fit_buffer]):
            self.lm.flush_buffers_to_file(path=output_dir)

        self.ctree_control = ctree_control

        pandas2ri.activate()
        train["response"] = train["response"].astype("category")
        r_train = pandas2ri.py2rpy(train)
        robjects.r.assign("train", r_train)  # type: ignore

        # import partykit and make objects
        importr("partykit")
        cforest = robjects.r["cforest"]
        ctree_control_func = robjects.r["ctree_control"]

        self.lm.add_fit_buffer("building forest...")
        self.lm.add_fit_buffer(f"number of trees = {ntree}")
        self.lm.add_fit_buffer(f"number of cores = {cores}")
        self.lm.add_fit_buffer(ctree_control.__str__())
        # define control options
        control = ctree_control_func(**dataclasses.asdict(ctree_control))  # type: ignore

        # build the tree
        start = time.perf_counter()
        self.fitted_model = cforest(  # type: ignore
            formula=robjects.r.formula("response ~ `.`"),  # type: ignore
            data=r_train,
            control=control,
            ntree=ntree,
            cores=cores,
        )
        end = time.perf_counter()

        elapsed_min = (end - start) / 60
        self.lm.add_fit_buffer(f"forest built in {elapsed_min:0.2f} minutes")

    def predict(self, newx, type: str):
        # activate pandas to R converter
        pandas2ri.activate()

        # convert newx to R object
        r_newx = pandas2ri.py2rpy(newx)
        # assign to variable in R environment
        robjects.r.assign("newx", r_newx)  # type: ignore
        # assign fitted model to variable in R environment
        robjects.r.assign("fitted_model", self.fitted_model)  # type: ignore
        robjects.r.assign("type", type)  # type: ignore

        # run predict and return the result
        pred = robjects.r("predict(fitted_model, newdata=newx, type=type)")

        if type == "response":
            robjects.r.assign("pred", pred)  # type: ignore
            pred = robjects.r("type.convert(pred, as.is=TRUE)")  # type: ignore
        return pred

    def assess(self, ytrue, ypredict, ypredict_probs, name: str = "clustermap"):
        # metrics
        _, _ = super().assess(ytrue, ypredict, ypredict_probs)

        # clustermap
        output_dir = Path(self.data.config["output"]["locations"]["cforest"])
        output_dir = output_dir.joinpath(self.runID, name)

        self._clustermap(ypredict_probs, ytrue, output_dir)

    def plot(self):
        raise NotImplementedError("No plot available for cforest")

    def save(self):
        # fetch the output dir for cforest
        output_dir = Path(self.data.config["output"]["locations"]["cforest"])
        # append the runID as a subfolder
        output_dir = output_dir.joinpath(self.runID, "cforest_model")

        # save the model as a pickle file
        super().save(path=output_dir)


# FUNCTIONS


def main():
    print("This file is meant to be imported, not run on it's own.")


if __name__ == "__main__":
    main()
