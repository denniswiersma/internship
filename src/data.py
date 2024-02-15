#!/usr/bin/env python3

"""
"""

# METADATA

# IMPORTS
import random
import time
from typing import Tuple, cast

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from sklearn.model_selection import train_test_split

from src.log_manager import LogManager

# CLASSES


class Data:
    def __init__(self, config: dict):
        """
        Initialises the Data class by using the data_folder to read the data.

        :param data_folder: Path to the folder containing the data.
        """
        self.config = config
        # as the data class is the default entry point, the log manager is
        # initialised here. The log manager is a singleton, so it will only be
        # initialised once after which it will be reused when the class is
        # instantiated again.
        self.lm = LogManager()

        self.read_data()

    def read_data(self):
        """
        Read the mixing matrix file and the tumor type annotation file.
        Only reads the "Name" and "TYPE3" columns in the mixing matrix.
        Renamees the "Unnamed: 0" column in the annotations to "samples"

        :param data_folder: Path to the folder containing the data.
        """
        self.lm.add_load_buffer("Loading data...")

        start = time.perf_counter()
        # read tumor type annotations
        with open(self.config["data"]["locations"]["tumor_types"]) as file:
            self.tumor_types: pd.DataFrame = pd.read_csv(
                file,
                sep="\t",
                usecols=[
                    self.config["data"]["columns"]["tumor_types"][
                        "sample_name"
                    ],
                    self.config["data"]["columns"]["tumor_types"]["response"],
                ],
            ).rename(
                columns={
                    self.config["data"]["columns"]["tumor_types"][
                        "sample_name"
                    ]: "samples",
                    self.config["data"]["columns"]["tumor_types"][
                        "response"
                    ]: "response",
                }
            )

        with open(self.config["data"]["locations"]["mixing_matrix"]) as file:
            self.mixing_matrix: pd.DataFrame = (
                pd.read_csv(
                    file,
                    sep="\t",
                )
                .rename(
                    columns={
                        self.config["data"]["columns"]["mixing_matrix"][
                            "sample_name"
                        ]: "samples"
                    }
                )
                .set_index("samples")
            )
        end = time.perf_counter()
        elapsed_sec = end - start
        self.lm.add_load_buffer(f"Data loaded in {elapsed_sec:0.4f} seconds")

    def filter_tt(self, min_n: int) -> None:
        """
        Filters both the mixing matrix and the tumor type annotations to only
        include tumor types with at least min_n samples.

        :param min_n: Minimum number of samples per tumor type
        :return: None
        """
        # get the number of samples per tumor type
        tt_counts = self.tumor_types["response"].value_counts()

        # get the tumor types with at least min_n samples
        tt_to_keep = tt_counts[tt_counts >= min_n].index

        # filter the tumor types
        self.tumor_types = self.tumor_types[
            self.tumor_types["response"].isin(tt_to_keep)
        ]

        # filter the mixing matrix
        self.mixing_matrix = self.mixing_matrix[
            self.mixing_matrix.index.isin(self.tumor_types["samples"])
        ]

    def replace_sample_sep(self, sep: str) -> None:
        """
        Replace the "-" in the sample names with another character in the
        mixing matrix and the tumor type annotations.

        :return: None
        """
        self.tumor_types["samples"] = self.tumor_types["samples"].str.replace(
            "-", sep
        )
        self.mixing_matrix.index = self.mixing_matrix.index.str.replace(
            "-", sep
        )

    def get_mm_with_tt(self) -> pd.DataFrame:
        """
        Returns the mixing matrix with a column called "TYPE3" containing each
        sample's tumor type.

        :return: Mixing matrix as a pandas dataframe
        """
        return self.tumor_types.merge(
            self.mixing_matrix,
            left_on="samples",
            right_on="samples",
            how="inner",
        ).set_index("samples")

    def get_r_mm_with_tt(self) -> pd.DataFrame:
        """
        Returns the mixing matrix as an R dataframe with a column called "TYPE3"
        containing each sample's tumor type.

        :return: Mixing matrix as an R dataframe
        """
        mm_with_tt = self.get_mm_with_tt()

        self.lm.add_load_buffer("Making R dataframe...")
        pandas2ri.activate()
        mm_with_tt["response"] = mm_with_tt["response"].astype("category")
        r_mm_with_tt = pandas2ri.py2rpy(mm_with_tt)
        ro.r.assign("r_mm_with_tt", r_mm_with_tt)  # type: ignore

        return r_mm_with_tt

    def get_train_test_val(
        self,
        data: pd.DataFrame,
        train_size: float | int,
        test_size: float | int,
        val_size: float | int,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits a pandas dataframe into test, train, and validation data using
        stratified sampling.

        :param data: A generic pandas dataframe. Any should work.
        :param train_size: Fraction of data to assign to train set.
        :param test_size: Fraction of data to assign to test set.
        :param val_size: Fraction of data to assign to validation set.
        :param seed: A seed for the random shuffling of data.
        the response variable column.
        :return: Three pandas dataframes in order of train, test, validate.
        """
        self.lm.add_prep_buffer(
            f"received splits\ntrain =\t{train_size}\ntest =\t{test_size}\nval =\t{val_size}"
        )
        size_list = [train_size, test_size, val_size]
        for i, size in enumerate(size_list):
            if isinstance(size, int):
                size_list[i] = size / len(data)

        if sum(size_list) != 1:
            raise ValueError(
                f"The sum of the size fractions is {sum(size_list)}, but it should be 1."
            )

        # split into train and temp data
        train_df, temp_df = train_test_split(
            data,
            test_size=(test_size + val_size),
            stratify=data["response"],
            random_state=seed,
        )

        temp_df = cast(pd.DataFrame, temp_df)

        test_df, val_df = train_test_split(
            temp_df,
            test_size=(val_size / (test_size + val_size)),
            stratify=temp_df["response"],
            random_state=seed,
        )

        train_df = cast(pd.DataFrame, train_df)
        test_df = cast(pd.DataFrame, test_df)
        val_df = cast(pd.DataFrame, val_df)

        self.lm.add_prep_buffer(
            f"split sizes\ntrain =\t{train_df.shape}\ntest =\t{test_df.shape}\nval =\t{val_df.shape}"
        )

        return train_df, test_df, val_df

    def split_xy(self, data: pd.DataFrame):
        # drop the response column to get just the features
        x = data.drop(columns=["response"])
        # select just the column with labels
        y = data["response"]

        return x, y

    def subset(
        self,
        data: pd.DataFrame,
        n_rows: int,
        n_cols: int,
        n_labels: int,
    ) -> pd.DataFrame:
        """
        Creates a randomly selected subset of the data with n_rows rows, n_cols
        columns, and n_labels labels.

        :param data: A pandas dataframe with the data to subset.
        :param n_rows: The number of rows to include in the subset.
        :param n_cols: The number of columns to include in the subset.
        :param n_labels: The number of labels to include in the subset.
        :return: A pandas dataframe with the subset.
        """
        # get a list of all unique labels
        unique_labels = data["response"].unique()

        # a list of n_labels randomly selected labels
        tumor_types = random.sample(unique_labels.tolist(), k=n_labels)

        # subset of data with only the randomly selected labels
        subset = data[data["response"].isin(tumor_types)]

        subset = subset.drop(columns=["response"])
        # randomly select n_rows rows and n_cols columns from the subset
        subset = subset.sample(n=n_rows, axis=0)
        subset = subset.sample(n=n_cols, axis=1)  # type: ignore

        subset.insert(loc=1, column="response", value=data["response"])
        # at least 2 samples per tumor type are needed for glmnet, so
        # if the number of samples per tumor type is less than 2
        # or if the number of tumor types is less than n_labels
        # recursively call this function until there are at least 2 samples per
        # tumor type and at least n_labels tumor types
        if (
            subset["response"].value_counts().min() < 2
            or len(subset["response"].unique()) < n_labels
        ):
            del subset  # delete last call's subset to preserve memory
            subset = self.subset(data, n_rows, n_cols, n_labels)

        return subset

    def subset_nrows_per_label(
        self,
        data: pd.DataFrame,
        nrows_per_label: int,
        ncols: int,
        nlabels: int,
    ) -> pd.DataFrame:
        """
        Creates a randomly selected subset of the data where each label has
        nrows_per_label rows, ncols columns, and nlabels labels.

        :param data: A pandas dataframe with the data to subset.
        :param nrows_per_label: The number of rows per label to include in the
        subset.
        :param ncols: The number of columns to include in the subset.
        :param nlabels: The number of labels to include in the subset.
        :return: A pandas dataframe with the subset.
        """
        # randomly select nrows_per_label rows for each label
        subset = data.groupby("response", as_index=False).apply(
            lambda x: x.sample(n=nrows_per_label, axis=0)
        )

        # pandas' groupby adds another index making a multiindex.
        # i think this is ridiculous behavior, but we have to deal with it.
        subset.reset_index(level=0, inplace=True)

        # randomly select ncols columns from the subset
        subset = subset.sample(n=ncols, axis=1)  # type: ignore

        # get a list of all unique labels
        unique_labels = subset["response"].unique()

        # a list of n_labels randomly selected labels
        tumor_types = random.sample(unique_labels.tolist(), k=nlabels)

        # subset of data with only the randomly selected labels
        subset = subset[subset["response"].isin(tumor_types)]

        return subset


# FUNCTIONS


def main():
    print("This file is meant to be imported, not run on it's own.")


if __name__ == "__main__":
    main()
