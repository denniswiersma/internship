#!/usr/bin/env python3

"""
Clustering analysis.
Cluster genes & cluster samples
then show annotation on the side (like a coloured bar for cancer types)
"""

import matplotlib.pyplot as plt
# IMPORTS
import pandas as pd
import seaborn as sns
from tqdm.contrib.concurrent import process_map

# CLASSES


class Mordor:
    def __init__(self, data_folder: str):
        self.data_folder = data_folder

        self.read_data(data_folder)

    def read_data(self, data_folder):
        """
        Open all three data files and store them in two class-level dataframes.

        Gene annotations, of which only the chromosome-gene mappings are of
        interest, are joined into the expressions dataframe.
        """
        # read expressions
        with open(
            data_folder
            + "CCLE_QCed_mRNA_NoDuplicates_CleanedIdentifiers_RMA-sketch.txt"
        ) as file:
            self.expressions = pd.read_csv(file, sep="\t")

        # read gene annotations, aka chromosome numbers
        with open(
            data_folder
            + "Genomic_Mapping_hgu133plus2_using_jetscore_3003201_v1.txt"
        ) as file:
            gene_annotation = pd.read_csv(file, sep="\t")
            self.expressions = (
                gene_annotation[["CHR_Mapping", "PROBESET"]]
                .join(other=self.expressions, on="PROBESET")
                .set_index("PROBESET")
            )

        # read tumor type annotations
        with open(data_folder + "CCLE__Sample_To_TumorType.csv") as file:
            self.tumor_type = pd.read_csv(file)

    def whos_that_pokemon(self, sample_name: str):
        """
        But with cancer.
        Returns a list of three cancer types for a given sample name.
        """
        return (
            self.tumor_type.loc[
                self.tumor_type["GSM_IDENTIFIER"] == sample_name
            ]
            .set_index("GSM_IDENTIFIER")
            .values.flatten()
        )

    def get_samples_from_tt(self, tt: str):
        """
        Returns a list of corresponding samples for a given tumor type.
        """
        return self.tumor_type.loc[self.tumor_type["TYPE"] == tt][
            "GSM_IDENTIFIER"
        ].to_list()

    def get_genes_from_chromosome(self, chromosome: int):
        """
        Returns the expressions dataframe containing only the data belonging
        to the corresponding chromosome
        """
        return self.expressions.loc[
            self.expressions["CHR_Mapping"] == chromosome
        ]

    def i_am_a_function(self, metric):
        sns.clustermap(self.expressions.iloc[:, 1:], metric=metric)
        plt.savefig(f"cluster_plots/{metric}.png")


# FUNCTIONS


def main():
    m = Mordor(data_folder="data/")

    methods = [
        "braycurtis",
        "canberra",
        "chebyshev",
        "cityblock",
        "correlation",
        "cosine",
        "dice",
        "euclidean",
        "hamming",
        "jaccard",
        "jensenshannon",
        "kulsinski",
        "mahalanobis",
        "matching",
        "minkowski",
        "rogerstanimoto",
        "russellrao",
        "seuclidean",
        "sokalmichener",
        "sokalsneath",
        "sqeuclidean",
        "yule",
    ]

    # process_map(
    #     m.i_am_a_function,
    #     methods,
    #     max_workers=25,
    #     desc="Applying all metrics",
    # )
    m.i_am_a_function("euclidean")


if __name__ == "__main__":
    main()
