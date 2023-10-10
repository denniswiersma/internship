#!/usr/bin/env python3

"""
"""

# METADATA

# IMPORTS
import pandas as pd
import numpy as np
from datetime import datetime
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from sklearn.model_selection import train_test_split

# CLASSES


class Data:
    def __init__(self, data_folder: str):
        """
        Initialises the Data class by using the data_folder to read the data.

        :param data_folder: Path to the folder containing the data.
        """
        self.data_folder = data_folder

        self.read_data(data_folder)

    def read_data(self, data_folder: str):
        """
        Read the mixing matrix file and the tumor type annotation file.
        Only reads the "Name" and "TYPE3" columns in the mixing matrix.
        Renamees the "Unnamed: 0" column in the annotations to "samples"

        :param data_folder: Path to the folder containing the data.
        """
        print("Reading data...")

        # read tumor type annotations
        with open(
            data_folder
            + "TCGA__Sample_To_TumorType_with_common_cancer_type_mapping_GEO_TCGA.csv"
        ) as file:
            self.tumor_types: pd.DataFrame = pd.read_csv(
                file, usecols=["Name", "TYPE3"]
            )

        with open(
            data_folder + "ica_flipped_mixing_matrix_consensus.tsv"
        ) as file:
            self.mixing_matrix: pd.DataFrame = (
                pd.read_csv(
                    file,
                    sep="\t",
                    # skiprows=lambda x: x not in self.tumor_types["Name"],
                )
                .rename(columns={"Unnamed: 0": "samples"})
                .set_index("samples")
            )

    def get_mm_with_tt(self) -> pd.DataFrame:
        """
        Returns the mixing matrix with a column called "TYPE3" containing each
        sample's tumor type.

        :return: Mixing matrix as a pandas dataframe
        """
        return self.tumor_types.merge(
            self.mixing_matrix, left_on="Name", right_on="samples", how="inner"
        )

    def get_r_mm_with_tt(self) -> pd.DataFrame:
        """
        Returns the mixing matrix as an R dataframe with a column called "TYPE3"
        containing each sample's tumor type.

        :return: Mixing matrix as an R dataframe
        """
        mm_with_tt = self.get_mm_with_tt()

        print("making R dataframe...")
        pandas2ri.activate()
        mm_with_tt["TYPE3"] = mm_with_tt["TYPE3"].astype("category")
        r_mm_with_tt = pandas2ri.py2rpy(mm_with_tt)
        ro.r.assign("r_mm_with_tt", r_mm_with_tt)

        return r_mm_with_tt

    def get_train_test_val(
        self,
        data: pd.DataFrame,
        train_size: float,
        test_size: float,
        val_size: float,
        seed: int = 42,
        stratify_col: str = "TYPE3",
    ) -> set[pd.DataFrame]:
        """
        Splits a pandas dataframe into test, train, and validation data using
        stratified sampling.

        :param data: A generic pandas dataframe. Any should work.
        :param train_size: Fraction of data to assign to train set.
        :param test_size: Fraction of data to assign to test set.
        :param val_size: Fraction of data to assign to validation set.
        :param seed: A seed for the random shuffling of data.
        :param stratify_col: The column to use for stratifying. Should be the
        the response variable column.
        :return: Three pandas dataframes in order of train, test, validate.
        """
        # split into train and temp data
        train_df, temp_df = train_test_split(
            data,
            test_size=(test_size + val_size),
            stratify=data[stratify_col],
            random_state=seed,
        )

        test_df, val_df = train_test_split(
            temp_df,
            test_size=(val_size / (test_size + val_size)),
            stratify=temp_df[stratify_col],
            random_state=seed,
        )

        return train_df, test_df, val_df

    def get_subset(
        self,
        data: pd.DataFrame,
        n_rows: int,
        n_cols: int,
        n_labels: int,
    ) -> pd.DataFrame:
        """
        Extract a subset of a pandas dataframe of a given size, with n number
        of response variables (labels).

        :param data: A generic pandas dataframe. Any should work.
        :param n_rows: Number of rows to select.
        :param n_cols: Number of columns to select.
        :param n_labels: Number of labels (reponse) to select.
        :raises ValueError: When you request more labels than are available.
        :return: The subsetted pandas dataframe.
        """
        # Get a list of unique labels in the "TYPE3" column
        unique_labels = data["TYPE3"].unique()

        # Ensure that there are enough unique labels to meet the requirement
        if len(unique_labels) < n_labels:
            raise ValueError("Not enough unique labels available.")

        # Randomly choose n_labels unique labels to include in the subset
        selected_labels = unique_labels[:n_labels]

        # Filter the DataFrame to include only rows with the selected labels
        filtered_data = data[data["TYPE3"].isin(selected_labels)]

        # Ensure n_rows and n_cols do not exceed the available rows and columns
        n_rows = min(n_rows, len(filtered_data))
        n_cols = min(n_cols, len(data.columns))

        # Select exactly n_rows rows and n_cols columns from the filtered subset
        subset = filtered_data.iloc[:n_rows, :n_cols]

        return subset


# FUNCTIONS


def main():
    now = datetime.now()
    print(f"starting time: {now.time()}")

    m = Data(data_folder="data/")

    print(m.mixing_matrix.shape)

    x = m.get_subset(
        data=m.get_mm_with_tt(), n_rows=5000, n_cols=50, n_labels=9, seed=80
    )

    print(x)


if __name__ == "__main__":
    main()
