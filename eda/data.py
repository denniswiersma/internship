#!/usr/bin/env python3

"""
"""

# METADATA

from datetime import datetime

import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
# IMPORTS
import pandas as pd
import seaborn as sns

# CLASSES


class Data:
    def __init__(self, data_folder: str):
        self.data_folder = data_folder

        self.read_data(data_folder)

    def read_data(self, data_folder):
        """
        Open all three data files and store them in two class-level dataframes.

        Gene annotations, of which only the chromosome-gene mappings are of
        interest, are joined into the expressions dataframe.
        """
        print("Reading data...")

        # read tumor type annotations
        with open(
            data_folder
            + "TCGA__Sample_To_TumorType_with_common_cancer_type_mapping_GEO_TCGA.csv"
        ) as file:
            self.tumor_type = pd.read_csv(file, usecols=["Name", "TYPE3"])

        with open(
            data_folder + "ica_flipped_mixing_matrix_consensus.tsv"
        ) as file:
            self.mixing_matrix = (
                pd.read_csv(
                    file,
                    sep="\t",
                    skiprows=lambda x: x not in self.tumor_type["Name"],
                )
                .rename(columns={"Unnamed: 0": "samples"})
                .set_index("samples")
            )

    def plot_stat(self, stat: str):
        match stat:
            case "min":
                data = self.mixing_matrix.min()
            case "max":
                data = self.mixing_matrix.max()
            case "mean":
                data = self.mixing_matrix.mean()
            case "median":
                data = self.mixing_matrix.median()
            case "sd":
                data = self.mixing_matrix.std()
            case "1q":
                data = self.mixing_matrix.quantile(q=0.25)
            case "3q":
                data = self.mixing_matrix.quantile(q=0.75)

        figure, bla = plt.subplots()

        # bla = sns.histplot(data, bins=25)
        bla.hist(data, edgecolor="black", bins=25)
        bla.set_title(
            f"Distribution of {stat} mixing matrix weights without samples with missing tumor types"
        )
        bla.set_xlabel("Mixing matrix weights")
        bla.set_ylabel("Frequency")

        plt.savefig(f"eda/plots/mm_{stat}_witout_missing_tt.png")
        plt.close()

    def heatmap(self):
        print("plotplotplot")

        # list all the unique tumor types found in the dataset
        unique_tumor_types = self.tumor_type["TYPE3"].unique()

        # pair each tumor type with a colour
        lut = dict(
            zip(
                unique_tumor_types,
                sns.color_palette("hls", len(unique_tumor_types)),
            )
        )

        # map each colour to its tumor type in the origional dataframe
        # this will result in a dataframe of coloyrs where the indexes match
        # with the indexes of the tumor types in the origional dataframe
        row_colours = (
            self.tumor_type["TYPE3"].rename({"TYPE3": "Tumor types"}).map(lut)
        )

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
        a, b = self.mixing_matrix.stack().idxmin()
        vmin = self.mixing_matrix.loc[[a], [b]].values

        # find the highest value in the dataframe
        # used for the upper limit of the colourmap
        a, b = self.mixing_matrix.stack().idxmax()
        vmax = self.mixing_matrix.loc[[a], [b]].values

        # normalise with a center of zero
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        # create the clustermap
        cm = sns.clustermap(
            self.mixing_matrix.reset_index(
                drop=True
            ),  # resetting the index needed to get the tumor type bar to colour
            method="ward",
            metric="correlation",
            row_colors=row_colours,  # adds tumor type bar
            xticklabels=False,
            yticklabels=False,
            cbar_kws={"label": "Weights"},  # adds label to cbar
            cmap="seismic",
            figsize=(10, 20),
            vmin=vmin,
            vmax=vmax,
            norm=norm,
        )

        # set x and y labels
        cm.ax_heatmap.set(xlabel="Transcriptional components", ylabel="Samples")

        # configure tumor type legend
        leg = cm.ax_heatmap.legend(
            loc="center right",
            bbox_to_anchor=(1.4, 0.8),
            handles=legend_TN,
            frameon=True,
        )
        leg.set_title(title="Tumor type", prop={"size": 10})

        cm.savefig("eda/plots/mm_heatmap.png")


# FUNCTIONS


def main():
    now = datetime.now()
    print(f"starting time: {now.time()}")

    m = Data(data_folder="data/")

    print(m.mixing_matrix.shape)

    stats = ["min", "max", "mean", "median", "sd", "1q", "3q"]

    # for stat in stats:
    #     m.plot_stat(stat)

    m.heatmap()


if __name__ == "__main__":
    main()
