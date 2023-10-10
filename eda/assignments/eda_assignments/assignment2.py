#!/usr/bin/env python3

"""
For every chromosome:
    median, mean, SD, 1q, 3q, min, max


Per sample returns some descriptive statistics for every chromosome
"""

# METADATA

# IMPORTS
import pandas as pd

# CLASSES


# FUNCTIONS
def main():
    # Read expression and gene annotation files
    with open(
        "CCLE_QCed_mRNA_NoDuplicates_CleanedIdentifiers_RMA-sketch.txt"
    ) as file:
        expressions = pd.read_csv(file, sep="\t")

    with open(
        "Genomic_Mapping_hgu133plus2_using_jetscore_3003201_v1.txt"
    ) as file:
        gene_annotation = pd.read_csv(file, sep="\t")

    # Join dataframes
    joined = (
        gene_annotation[["CHR_Mapping", "PROBESET"]]
        .join(other=expressions, on="PROBESET")
        .set_index("PROBESET")
    )

    # Set table width when printing to cli
    # pd.set_option('display.max_columns', 8)

    # Grouping by chromosome and calculate summary statistics
    print(joined.groupby("CHR_Mapping").describe().T)

    # print(joined.groupby("CHR_Mapping").apply(pd.Series.describe, axis=1))


if __name__ == "__main__":
    main()
