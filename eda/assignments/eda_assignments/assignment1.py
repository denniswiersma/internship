#!/usr/bin/env python3

"""
For every chromosome, compare the distribution of expression levels of genes
in cancer type A vs cancer type B.
A, B for all cancer types.
(histogram), (...)
"""

# METADATA

# IMPORTS
import os
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import multiprocessing as mp
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

    def compare_expression_plot(
        self,
        the_list: list[int, str, str],
    ):
        """
        Produces two histograms comparing the distribution of expression
        levels between tumor type A (tt_A) & tumor type B (tt_B)
        """
        chromosome = the_list[0]
        tt_A = the_list[1]
        tt_B = the_list[2]

        chr_filtered_data = self.get_genes_from_chromosome(chromosome)

        # select the data belonging to each respective tumor type
        a_df = chr_filtered_data[self.get_samples_from_tt(tt_A)]
        b_df = chr_filtered_data[self.get_samples_from_tt(tt_B)]

        # stop the functino if either of the dataframes is empty
        if a_df.empty or b_df.empty:
            return

        # initialise empty lists to hold expression levels
        a_list = []
        b_list = []

        # fetch all expression levels from their dataframes and transfer
        # them to their respective lists, effectively flattening the datafram.

        # not doing this will give each of the histogram's bars n_samples
        # number of sub-bars
        for col in a_df.columns:
            a_list.extend(a_df[col].tolist())

        for col in b_df.columns:
            b_list.extend(b_df[col].tolist())

        # initialise a 1x2 figure that share y axes
        figure, (axes_A, axes_B) = plt.subplots(1, 2, sharey=True)

        # draw histograms and their labels to subplots
        axes_A.hist(a_list, bins=10, density=True, edgecolor="black")
        axes_A.set_title(tt_A)

        axes_B.hist(b_list, bins=10, density=True, edgecolor="black")
        axes_B.set_title(tt_B)

        # draw figure wide labels
        figure.suptitle(
            f"Distribution of expression levels between tumor types in chromosome {chromosome}"
        )
        figure.supxlabel("Expression levels")
        figure.supylabel("Density")

        plt.savefig(f"plots/{chromosome}_{tt_A}_{tt_B}.png")
        plt.close()


# FUNCTIONS


def main():
    m = Mordor(data_folder="data/")

    # find all the unique tumor types
    unique_tt = {
        tt
        for col_name in ["TYPE", "Type2", "TYPE3"]
        for tt in m.tumor_type[col_name].tolist()
    }

    # generate all combinations of tumor type pairs
    tt_combos = itertools.combinations(unique_tt, 2)

    # generate a list of jobs containing argument lists (chr, tt_A, tt_B)
    jobs = []
    for combo in tt_combos:
        for chr in m.expressions["CHR_Mapping"].unique():
            jobs.append([chr, *combo])

    # multiprocessed plot making
    process_map(
        m.compare_expression_plot,
        jobs,
        max_workers=mp.cpu_count(),
        chunksize=100,
        desc="Comparing cancer types & plotting results",
    )

    print(f"Number of plots generated: {len(os.listdir('plots/'))}")


if __name__ == "__main__":
    main()
