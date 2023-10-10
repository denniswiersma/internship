#!/usr/bin/env python3

"""
"""

# METADATA

# IMPORTS
import tomllib
from enum import Enum
import pandas as pd
from data import Data

import matplotlib.pyplot as plt
import seaborn as sns
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2 import robjects


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

    def fit_glm(self, train: pd.DataFrame):
        # import glmnet
        importr("glmnet")
        # assign r callable to python variable
        cv_glmnet = robjects.r["cv.glmnet"]

        # enum representing alpha options
        class Alpha(Enum):
            lasso = 1
            elastic_net = 0.5
            ridge = 0

        # select all columns but the one with labels
        xtrain = train.loc[:, train.columns != "response"]
        # take a subset (columns 1 through 5) of the data because time
        xtrain = xtrain.iloc[:, 1:]
        # last row is a weird NaN value that shouldn't be there, so remove it
        ytrain = train.iloc[:, :]
        # select just the column with labels
        ytrain = ytrain.loc[:, ytrain.columns == "response"]

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
        # cv_glmnet_res = cv_glmnet(
        #     r_xtrain,
        #     r_ytrain,
        #     family="multinomial",
        #     intercept=True,
        #     standardize=False,
        # )

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
    with open("config.toml", "rb") as file:
        config = tomllib.load(file)

    data = Data(config)

    glm = GLM(data)

    mm_with_tt = data.get_mm_with_tt()

    subset = data.get_subset(mm_with_tt, n_rows=10000, n_cols=100, n_labels=10)
    subset = subset.set_index("samples")
    print(subset)

    train, test, val = data.get_train_test_val(
        train_size=0.7, test_size=0.15, val_size=0.15, data=subset
    )
    # glm.plot_label_distribution(train, test, val)

    glm.fit_glm(train)


if __name__ == "__main__":
    main()
