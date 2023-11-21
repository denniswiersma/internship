#!/usr/bin/env python3

"""
"""

# METADATA

# IMPORTS
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from src.data import Data
from src.ml.model import Model


# CLASSES
class GLM(Model):
    def __init__(self, data: Data):
        """
        Initialises the GLM class by setting the mixing matrix, tumor types,
        and mixing matrix with tumor types to the values in the given instance
        of the Data class.

        :param data: The instance of the Data class to use.
        """
        self.data: Data = data
        self.mm_with_tt: pd.DataFrame = data.get_mm_with_tt()

    def plot_label_distribution(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots the distribution of the cancer type labels in the train, test,
        and validation sets.

        :param train_df: The training set.
        :param test_df: The test set.
        :param val_df: The validation set
        :return: The figure and axes of the plot.
        """
        # Create subplots for each dataset
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        unique_tt = train_df["response"].unique()

        # Plot the label distribution for the training set
        sns.countplot(data=train_df, x="response", ax=axes[0], order=unique_tt)
        axes[0].set_title("Training Set Label Distribution")
        axes[0].set_xlabel("Label")
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis="x", rotation=90)

        # Plot the label distribution for the test set
        sns.countplot(data=test_df, x="response", ax=axes[1], order=unique_tt)
        axes[1].set_title("Test Set Label Distribution")
        axes[1].set_xlabel("Label")
        axes[1].set_ylabel("Count")
        axes[1].tick_params(axis="x", rotation=90)

        # Plot the label distribution for the validation set
        sns.countplot(data=val_df, x="response", ax=axes[2], order=unique_tt)
        axes[2].set_title("Validation Set Label Distribution")
        axes[2].set_xlabel("Label")
        axes[2].set_ylabel("Count")
        axes[2].tick_params(axis="x", rotation=90)

        plt.tight_layout()
        plt.show()

        return fig, axes  # type: ignore

    def fit(
        self,
        xtrain: pd.DataFrame,
        ytrain: pd.DataFrame,
        alpha: float = 0.5,
        thresh: float = 1e-7,
        maxit: int = int(1e5),
    ) -> robjects.r:  # type: ignore
        # import glmnet into R environment
        """
        Runs cv.glmnet and glmnet on the given dataset.

        :param xtrain: The covariates in the training set.
        :param ytrain: The response in the training set.
        :param alpha: The alpha value to control regularization in glmnet.
        :param thresh: The threshold value to control convergence in cv.glmnet.
        :param maxit: The maximum number of iterations to run in glmnet.
        :return: The fitted model as an R object.
        """
        importr("glmnet")

        # convert python dataframes to R objects
        # activate converter
        pandas2ri.activate()
        # convert to R dataframes
        r_xtrain = pandas2ri.py2rpy(xtrain)
        r_ytrain = pandas2ri.py2rpy(ytrain)
        # assign to variables in R environment
        robjects.r.assign("xtrain", r_xtrain)  # type: ignore
        robjects.r.assign("ytrain", r_ytrain)  # type: ignore
        # remove missing values
        robjects.r("xtrain <- na.omit(xtrain)")
        robjects.r("ytrain <- na.omit(ytrain)")
        # convert x to matrix
        robjects.r("xtrain <- data.matrix(xtrain)")
        # convert y to factor
        robjects.r("ytrain <- factor(unlist(ytrain))")

        print("Doing cv.glmnet...")
        # assign thresh to variable in R environment
        robjects.r.assign("thresh", thresh)  # type: ignore

        # run cv.glmnet
        cv_glmnet_res = robjects.r(
            "cv.glmnet(x=xtrain, y=ytrain, family='multinomial', intercept=TRUE, standardize=FALSE, thresh=thresh)"
        )

        # assign cv_glmnet_res, maxit, and alpha to variables in R environment
        robjects.r.assign("cv_glmnet_res", cv_glmnet_res)  # type: ignore
        robjects.r.assign("maxit", maxit)  # type: ignore
        robjects.r.assign("alpha", alpha)  # type: ignore

        print(cv_glmnet_res)
        print("Doing glmnet...")

        # run glmnet
        fit_optimised = robjects.r(
            "glmnet(x=xtrain, y=ytrain, alpha=alpha, lambda=cv_glmnet_res$lambda.min, family='multinomial', intercept=TRUE, standardize=FALSE, maxit=maxit)"
        )

        print(fit_optimised)

        self.runID = self._generate_runID()
        self.fitted_model = fit_optimised

    def predict(self, newx: pd.DataFrame, type: str):
        """
        Predicts the response for the given covariates using the given model by
        utilising the predict function in R.

        :param model: The model to use for the prediction.
        :param newx: A dataframe with the covariates for which to predict the
        response.
        :param type: The type of prediction to make. Usually "class" or "prob".
        """
        # activate the pandas to R converter
        pandas2ri.activate()

        # convert the covariates to an R dataframe
        r_newx = pandas2ri.py2rpy(newx)
        # assign to variable in R environment
        robjects.r.assign("newx", r_newx)  # type: ignore
        # turn newx into a matrix
        robjects.r("newx <- data.matrix(newx)")

        # assign model and type to variables in R environment
        robjects.r.assign("model", self.fitted_model)  # type: ignore
        robjects.r.assign("type", type)  # type: ignore

        # run predict and return the result
        return robjects.r("predict(model, newx=newx, type=type)")

    def assess(
        self, ytrue: pd.Series, ypredict: list, ypredict_probs: list[list]
    ):
        _, _ = super().assess(ytrue, ypredict, ypredict_probs)

        # clustermap
        output_dir = Path(self.data.config["output"]["locations"]["glmnet"])
        output_dir = output_dir.joinpath(self.runID)

        file_name = "clustermap"
        output_dir = output_dir.joinpath(file_name)

        self._clustermap(ypredict_probs, ytrue, output_dir)

    def plot(self):
        output_dir = Path(self.data.config["output"]["locations"]["glmnet"])
        output_dir = output_dir.joinpath(self.runID)

        file_name = "glmnet"
        output_dir = output_dir.joinpath(file_name)
        output_dir = output_dir.with_suffix(".png")
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        grdevices = importr("grDevices")
        grdevices.png(output_dir.as_posix(), width=512, height=512)

        robjects.r.plot(self.fitted_model, xvar="lambda", label=True)  # type: ignore

        grdevices.dev_off()
        print(f"Plot saved to {output_dir}")

    def save(self):
        output_dir = Path(self.data.config["output"]["locations"]["glmnet"])
        output_dir = output_dir.joinpath(self.runID)

        file_name = "glmnet"
        output_dir = output_dir.joinpath(file_name)

        super().save(output_dir)


# FUNCTIONS


def main():
    print("This file is meant to be imported, not run on it's own.")


if __name__ == "__main__":
    main()
