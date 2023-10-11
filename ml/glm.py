#!/usr/bin/env python3

"""
"""

# METADATA

# IMPORTS
from enum import Enum

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from data import Data


# CLASSES
class GLM:
    def __init__(self, data: Data):
        self.mixing_matrix: pd.DataFrame = data.mixing_matrix
        self.tumor_types: pd.DataFrame = data.tumor_types
        self.mm_with_tt: pd.DataFrame = data.get_mm_with_tt()

    def plot_label_distribution(self, train_df, test_df, val_df) -> None:
        # Create subplots for each dataset
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the label distribution for the training set
        sns.countplot(data=train_df, x="response", ax=axes[0])
        axes[0].set_title("Training Set Label Distribution")
        axes[0].set_xlabel("Label")
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis="x", rotation=90)

        # Plot the label distribution for the test set
        sns.countplot(data=test_df, x="response", ax=axes[1])
        axes[1].set_title("Test Set Label Distribution")
        axes[1].set_xlabel("Label")
        axes[1].set_ylabel("Count")
        axes[1].tick_params(axis="x", rotation=90)

        # Plot the label distribution for the validation set
        sns.countplot(data=val_df, x="response", ax=axes[2])
        axes[2].set_title("Validation Set Label Distribution")
        axes[2].set_xlabel("Label")
        axes[2].set_ylabel("Count")
        axes[2].tick_params(axis="x", rotation=90)
        plt.tight_layout()
        plt.show()

    def fit_glm(self, xtrain: pd.DataFrame, ytrain: pd.DataFrame):
        # import glmnet
        importr("glmnet")
        # assign r callable to python variable
        cv_glmnet = robjects.r["cv.glmnet"]

        # enum representing alpha options
        class Alpha(Enum):
            lasso = 1
            elastic_net = 0.5
            ridge = 0

        # convert python variables to R objects
        pandas2ri.activate()
        r_xtrain = pandas2ri.py2rpy(xtrain)
        r_ytrain = pandas2ri.py2rpy(ytrain)
        robjects.r.assign("xtrain", r_xtrain)
        robjects.r.assign("ytrain", r_ytrain)
        # change ytrain to a factor
        robjects.r("xtrain <- na.omit(xtrain)")
        robjects.r("ytrain <- na.omit(ytrain)")
        robjects.r("xtrain <- data.matrix(xtrain)")
        robjects.r("ytrain <- factor(unlist(ytrain))")

        # convert R object back to python
        r_xtrain = robjects.r["xtrain"]
        r_ytrain = robjects.r["ytrain"]

        print("Doing cv.glmnet...")

        cv_glmnet_res = robjects.r(
            "cv.glmnet(x=xtrain, y=ytrain, family='multinomial', intercept=TRUE, standardize=FALSE)"
        )

        robjects.r.assign("cv_glmnet_res", cv_glmnet_res)
        robjects.r.assign("alpha", 1)

        print("Doing glmnet...")

        fit_optimised = robjects.r(
            "glmnet(x=xtrain, y=ytrain, alpha=alpha, lambda=cv_glmnet_res$lambda.min, family='multinomial', intercept=TRUE, standardize=FALSE, maxit=1e+06)"
        )

        print(fit_optimised)


# FUNCTIONS


def main():
    print("This file is meant to be imported, not run on it's own.")


if __name__ == "__main__":
    main()
