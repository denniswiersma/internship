#!/usr/bin/env python3

"""
"""

# METADATA

# IMPORTS
import pandas as pd
from data import Data
from scipy import stats
import itertools
import tqdm
from functools import partial
import multiprocessing as mp
import seaborn as sns
import matplotlib.pyplot as plt
import re


# CLASSES
class StatsCalculator:
    def __init__(self, expressions: pd.DataFrame):
        print("Preparing stat calculations...")
        self.expressions: pd.DataFrame = expressions.iloc[:, 1:]
        self.col_iter = [
            self.expressions[col] for col in self.expressions.columns
        ]
        self.col_combos = itertools.product(self.expressions.columns, repeat=2)

    def check_norm(self, sig: float = 0.05):
        print("Testing for normality in all samples...")
        res = self.expressions.apply(stats.anderson)
        res.index = ["statistic", "critical_values", "significance"]
        res = res.T

        # styling table and write to latex
        table_subset = res.iloc[0:10]
        table_subset.rename(
            columns={"critical_values": "criticalvalues"}, inplace=True
        )

        self.style_table(
            table_subset,
            save_loc="documents/log/tables/norm.tex",
            max_axis_len=14,
        )

        match sig:
            case 0.15:
                sig_index = 0
            case 0.10:
                sig_index = 1
            case 0.05:
                sig_index = 2
            case 0.025:
                sig_index = 3
            case 0.001:
                sig_index = 4

        res["critical_values"] = [
            critical_values[sig_index]
            for critical_values in res["critical_values"]
        ]

        res["significance"] = [
            significance[sig_index] for significance in res["significance"]
        ]

        print(res[res["statistic"] < res["critical_values"]])

        # n_norm = res.where(res["statistic"] < res["critical_values"]).shape[0]
        n_norm = res[res["statistic"] < res["critical_values"]].shape[0]
        n_not_norm = res.shape[0] - n_norm

        return res, n_norm, n_not_norm

    def anova(self):
        print("Performing ANOVA over all samples...")
        f, p = stats.f_oneway(*self.col_iter)
        print("H0: population mean of all groups are equal.")

        return f, p

    def kruskal(self):
        print("Performing Kruskal-Wallis over all samples...")
        f, p = stats.kruskal(*self.col_iter)
        print("H0: population median of all groups are equal.")

        return f, p

    def t_test(self, is_normal: bool = True):
        print("Performing t tests for all sample combinations...")
        print("WARNING: loading bar moves at start of, not end of, iteration")

        if is_normal:
            func = partial(stats.ttest_ind, equal_var=False)
        else:
            func = stats.mannwhitneyu

        t_tests = [
            [self.expressions[x], self.expressions[y]]
            for x, y in self.col_combos
        ]

        with mp.Pool(mp.cpu_count()) as p:
            results = p.starmap(
                func,
                tqdm.tqdm(t_tests, total=len(t_tests)),
                chunksize=100_000,
            )

        data_dict = {}

        for sample_pair, result in zip(
            itertools.product(self.expressions.columns, repeat=2), results
        ):
            sample1, sample2 = sample_pair
            data_dict.setdefault(sample2, {})[sample1] = result.pvalue

        df = pd.DataFrame(data_dict)

        return df

    def fisher(self, g1: str, g2: str, group_threshold: float):
        print(f"Performing fisher exact test for {g1} vs {g2}...")
        g1_data = self.expressions.loc[[g1]].squeeze()
        g2_data = self.expressions.loc[[g2]].squeeze()

        g1_nh, g1_nl = len(g1_data[g1_data > group_threshold]), len(
            g1_data[g1_data < group_threshold]
        )
        g2_nh, g2_nl = len(g2_data[g2_data > group_threshold]), len(
            g2_data[g2_data < group_threshold]
        )

        count_table = pd.DataFrame(
            data=[[g1_nh, g1_nl], [g2_nh, g2_nl]],
            index=[g1, g2],
            columns=[f"> {group_threshold}", f"< {group_threshold}"],
        )

        count_table = count_table.T
        f, p = stats.fisher_exact(count_table)

        self.style_table(count_table, "documents/log/tables/fisher.tex", 14)

        return f, p, count_table

    def style_table(
        self, dataframe: pd.DataFrame, save_loc: str, max_axis_len: int
    ):
        table_subset = dataframe.iloc[0:10]
        table_subset.index = [
            re.sub(pattern="_", repl="\_", string=index_name[-max_axis_len:])
            for index_name in table_subset.index
        ]

        table_subset.columns = [
            re.sub(pattern="_", repl="\_", string=column_name[-max_axis_len:])
            for column_name in table_subset.columns
        ]
        style = table_subset.style
        style.applymap_index(lambda v: "font-weight: bold;", axis="index")
        style.applymap_index(lambda v: "font-weight: bold;", axis="columns")

        with open(save_loc, "w") as file:
            style.to_latex(
                buf=file,
                convert_css=True,
                column_format="|l|l|l|l|",
                clines="skip-last;data",
                hrules=True,
            )


def main():
    data = Data(data_folder="data/")
    sts = StatsCalculator(data.expressions)

    data_normality, n_norm, n_not_norm = sts.check_norm(sig=0.05)
    print(f"Out of {n_norm + n_not_norm} samples:")
    print(f"{n_norm} are normally distributed\n{n_not_norm} are not\n")

    f_anova, p_anova = sts.anova()
    print(f"F statistic: {f_anova}\np value: {p_anova}\n")

    f_kruskal, p_kruskal = sts.kruskal()
    print(f"F statistic: {f_kruskal}\np value: {p_kruskal}\n")

    # t_test = sts.t_test(is_normal=True)
    # print(t_test)
    # sns.heatmap(t_test)
    # plt.show()

    f_fisher, p_fisher, count_table = sts.fisher(
        "1560957_at", "205483_s_at", 3.5
    )
    print(count_table)
    print(f"Statistic: {f_fisher}\np value: {p_fisher}\n")


if __name__ == "__main__":
    main()
