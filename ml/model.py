#!/usr/bin/env python3

"""
An abstract base class that defines an interface for wrapper classes that
implement the usage of machine learning libraries. Subclasses provide, among
other things, an incapsulation of the complexities of the underlying library's
API, and a standardised interface for the analysis stage of this project.

Implementations are expected to override the abstract methods, and provide
additional case-specific methods as needed.
"""

# METADATA

# IMPORTS
import pickle as pkl
import secrets
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from rpy2 import robjects
from sklearn import metrics


# CLASSES
class Model(ABC):
    @abstractmethod
    def __init__(self):
        ...

    @staticmethod
    def _generate_runID() -> str:
        now = datetime.now()
        return now.strftime("%Y%m%d%H%M%S") + "_" + secrets.token_hex(3)

    @abstractmethod
    def fit(self):
        self.fitted_model = self.fit()
        self.runID = self._generate_runID()

    @abstractmethod
    def predict(self, newx, type: str):
        ...

    @abstractmethod
    def assess(self, ytrue, ypredict, ypredict_probs):
        # metrics
        # handle all metrics centrally to allow for easy additions
        aucroc = (
            metrics.roc_auc_score(ytrue, ypredict_probs, multi_class="ovr"),
        )
        mcc = metrics.matthews_corrcoef(ytrue, ypredict)
        print(f"AUC-ROC: {aucroc[0]}\nMCC: {mcc}")

        # clustermap code goes here

        return aucroc[0], mcc

    def _clustermap(
        self, ypredict_probs: list[list], ytest: pd.Series, path: Path
    ):
        """
        Plots a clustermap of the predictions to visualise the probability of
        each tumor type for each sample.

        :param model: The model used to make the predictions.
        :param ypredict_probs: The predicted probabilities.
        :param ytest: The true response for the testing data.
        """
        robjects.r.assign("response", ytest)  # type: ignore
        # convert the predictions to a dataframe with the correct column names
        predictions = pd.DataFrame(
            ypredict_probs, columns=robjects.r("levels(as.factor(response))")  # type: ignore
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

        path = path.with_suffix(".png")
        path.parent.mkdir(parents=True, exist_ok=True)

        cm.savefig(path, bbox_inches="tight")
        print(f"Clustermap saved to {path}")

    @abstractmethod
    def plot(self):
        ...

    @abstractmethod
    def save(self, path: Path):
        path = path.with_suffix(".pkl")
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as file:
            pkl.dump(self.fitted_model, file=file)
        print(f"Model saved to {path}")


# FUNCTIONS


def main():
    print("This file is meant to be imported, not run on it's own.")


if __name__ == "__main__":
    main()
