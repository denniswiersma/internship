#!/usr/bin/env python3

"""
"""

# METADATA

# IMPORTS
from typing import Tuple

import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn import metrics
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay

from data import Data


# CLASSES
class GLM:
    def __init__(self, data: Data):
        """
        Initialises the GLM class by setting the mixing matrix, tumor types,
        and mixing matrix with tumor types to the values in the given instance
        of the Data class.

        :param data: The instance of the Data class to use.
        """
        self.mixing_matrix: pd.DataFrame = data.mixing_matrix
        self.tumor_types: pd.DataFrame = data.tumor_types
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
        return fit_optimised

    def predict(self, model, newx: pd.DataFrame, type: str):
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
        robjects.r.assign("model", model)  # type: ignore
        robjects.r.assign("type", type)  # type: ignore

        # run predict and return the result
        return robjects.r("predict(model, newx=newx, type=type)")

    def assess(
        self, ytrue: pd.Series, ypredict: list, ypredict_probs: list[list]
    ):
        plt.figure(figsize=(15, 15))

        ConfusionMatrixDisplay.from_predictions(
            ytrue,
            ypredict,
            normalize="true",
            cmap="binary",
            xticks_rotation="vertical",
        )
        plt.show()

        # print the classification report containing precision, recall, and f1
        print(metrics.classification_report(ytrue, ypredict))

        # print the AUC ROC
        print(
            "AUC ROC:",
            metrics.roc_auc_score(ytrue, ypredict_probs, multi_class="ovr"),
        )
        # print the MCC
        print("MCC:", metrics.matthews_corrcoef(ytrue, ypredict))

    def assess_cm(self, ypredict_probs: list[list], ytest: pd.Series):
        """
        Plots a clustermap of the predictions to visualise the probability of
        each tumor type for each sample.

        :param model: The model used to make the predictions.
        :param ypredict_probs: The predicted probabilities.
        :param ytest: The true response for the testing data.
        """
        print(type(ypredict_probs))
        # convert the predictions to a dataframe with the correct column names
        predictions = pd.DataFrame(
            ypredict_probs, columns=robjects.r("names(coef(model))")  # type: ignore
        )

        # list all the unique tumor types found in the dataset
        unique_tt = ytest.unique()

        # pair each tumor type with a colour
        lut = dict(
            zip(
                unique_tt,
                sns.color_palette("hls", len(unique_tt)),
            )
        )

        # map each colour to its tumor type in the origional dataframe
        # this will result in a dataframe of coloyrs where the indexes match
        # with the indexes of the tumor types in the origional dataframe
        row_colours = ytest.map(lut)

        # make patchesfor the tumor type legend: colour and tumor type label
        legend_TN = [
            mpatches.Patch(facecolor=colour, label=label, edgecolor="black")
            for label, colour in lut.items()
        ]

        legend_TN.append(
            mpatches.Patch(
                facecolor="white",
                label="Missing cancer type",
                edgecolor="black",
            )
        )

        # find the lowest value in the dataframe
        # used for the lower limit of the colourmap
        print(type(predictions.stack().idxmin()))
        a, b = predictions.stack().idxmin()  # type: ignore
        vmin = predictions.loc[[a], [b]].values

        # find the highest value in the dataframe
        # used for the upper limit of the colourmap
        a, b = predictions.stack().idxmax()  # type: ignore
        vmax = predictions.loc[[a], [b]].values

        # normalise with a center of zero
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.5, vmax=vmax)

        # create the clustermap
        cm = sns.clustermap(
            predictions.reset_index(
                drop=True
            ),  # resetting the index needed to get the tumor type bar to colour
            method="ward",
            metric="euclidean",
            row_colors=row_colours.reset_index(
                drop=True
            ),  # adds tumor type bar
            xticklabels=True,
            yticklabels=False,
            cbar_kws={"label": "Probability"},  # adds label to cbar
            cmap="seismic",
            figsize=(10, 20),
            vmin=vmin,
            vmax=vmax,
            norm=norm,
        )

        # set x and y labels
        cm.ax_heatmap.set(xlabel="Predicted label", ylabel="Samples")

        # configure tumor type legend
        leg = cm.ax_heatmap.legend(
            loc="center right",
            bbox_to_anchor=(1.4, 0.8),
            handles=legend_TN,
            frameon=True,
        )
        leg.set_title(title="Tumor type", prop={"size": 10})
        plt.show()


# FUNCTIONS


def main():
    print("This file is meant to be imported, not run on it's own.")


if __name__ == "__main__":
    main()
