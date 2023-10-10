#!/usr/bin/env python3

"""
"""

# METADATA

import matplotlib.pyplot as plt
# IMPORTS
import pandas as pd

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

    def plot_sample(self, *, sample_name: str = None, ncol: int = None):
        figure, axes = plt.subplots()

        if sample_name and ncol:
            raise TypeError("sample_name and ncol are mutually exclusive")
        elif ncol:
            sample_name = self.expressions.columns[ncol]

        axes.hist(self.expressions[sample_name], edgecolor="black")
        axes.set_title(f"Expression levels of sample: {sample_name}")
        axes.set_xlabel("Expression Level")
        axes.set_ylabel("Frequency")

        plt.show()


# FUNCTIONS


def main():
    m = Data(data_folder="data/")

    print(m.expressions)

    m.plot_sample(ncol=1)


if __name__ == "__main__":
    main()
