    #!/usr/bin/env python3.9
# coding=utf-8
import pickle
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as nppip
import os
import matplotlib.pyplot as plt
import datetime

regions = ["PHA", "JHM", "STC", "ZLK"]

types_of_crashes = [
    "ina komunikacia",
    "dvojpruhová komunikácia",
    "trojpruhová komunikácia",
    "štvorpruhová komunikácia",
    "viacpruhová komunikácia",
    "rychlostní komunikace",
]

colors = ["slategray", "lightsteelblue", "cornflowerblue", "royalblue"]

cases_weather = [
    "neztížené",
    "mlha",
    "na počátku deště, slabý déšť, mrholení apod.",
    "déšť",
    "sněžení",
    "tvoří se námraza, náledí",
    "nárazový vítr (boční, vichřice apod.)",
]

MB_IN_B = 1048576
# muzete pridat libovolnou zakladni knihovnu ci knihovnu predstavenou na prednaskach
# dalsi knihovny pak na dotaz

""" Ukol 1:
načíst soubor nehod, který byl vytvořen z vašich dat. Neznámé integerové hodnoty byly mapovány na -1.

Úkoly:
- vytvořte sloupec date, který bude ve formátu data (berte v potaz pouze datum, tj sloupec p2a)
- vhodné sloupce zmenšete pomocí kategorických datových typů. Měli byste se dostat po 0.5 GB. Neměňte však na kategorický typ region (špatně by se vám pracovalo s figure-level funkcemi)
- implementujte funkci, která vypíše kompletní (hlubkou) velikost všech sloupců v DataFrame v paměti:
orig_size=X MB
new_size=X MB

Poznámka: zobrazujte na 1 desetinné místo (.1f) a počítejte, že 1 MB = 1e6 B. 
"""


def make_folder_if_not_exist(folder):
    folder_to_make = os.getcwd() + "/" + folder

    if not os.path.exists(folder_to_make):
        os.makedirs(folder_to_make)


def get_dataframe(filename: str, verbose: bool = False) -> pd.DataFrame:
    df = pd.read_pickle(filename)

    data = pd.DataFrame.from_dict(df)

    if verbose:
        value = data.memory_usage(deep=True).sum() / MB_IN_B
        print("orig_size={} MB".format(str(round(value, 1))))

    # changing types of some object types to category
    column_to_change = ["g", "h", "k", "p", "t", "s", "o", "j", "i"]
    data[column_to_change] = data[column_to_change].apply(
        lambda x: x.astype("category")
    )

    # rename and change type to date
    data["p2a"] = data["p2a"].astype("datetime64")
    data.rename(columns={"p2a": "date"}, inplace=True)

    if verbose:
        value = data.memory_usage(deep=True).sum() / MB_IN_B
        print("new_size={} MB".format(str(round(value, 1))))

    return data


# Ukol 2: počty nehod v jednotlivých regionech podle druhu silnic


def plot_roadtype(
    df: pd.DataFrame, fig_location: str = None, show_figure: bool = False
):

    # change numbers by name of accident
    if df is None:
        df = get_dataframe()
    df["p21"] = pd.cut(
        df["p21"], [-1, 0, 1, 2, 4, 5, 6], labels=types_of_crashes, ordered=False
    )

    df = (
        df.loc[df["region"].isin(regions)][["region", "p21", "p1"]]
        .groupby(["region", "p21"])
        .agg({"p1": "count"})
        .reset_index()
    )

    figure, axes = plt.subplots(
        nrows=2, ncols=3, constrained_layout=True, figsize=(10, 10)
    )

    sns.set_palette(colors)

    figure.suptitle("Druhy silnic", fontsize=16)

    counter = 0

    for el in types_of_crashes:

        ax = sns.barplot(
            ax=axes[int(counter - 3 >= 0)][counter % 3],
            data=df.loc[df["p21"] == el],
            x="region",
            y="p1",
        )
        ax.set_facecolor("lavender")
        ax.set_title(el)

        if counter - 3 < 0:
            ax.axes.xaxis.set_visible(False)

        if counter == 0 or counter == 3:
            ax.set_ylabel("Pocet nehod")
        else:
            ax.axes.yaxis.label.set_visible(False)

        counter += 1

    if fig_location is not None:
        dir = fig_location.rsplit("/", 1)
        if len(dir) > 1:
            make_folder_if_not_exist(dir[0])
        plt.savefig(fig_location)

    if show_figure:
        plt.show()


# Ukol3: zavinění zvěří
def plot_animals(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):

    if df is None:
        df = get_dataframe()

    causes = ["řidičem", "iné", "zvěří"]

    df = df.loc[((df["region"].isin(regions)) & (df["p58"] == 5))].copy()

    df["p10"] = pd.cut(df["p10"], [0, 2, 3, 4], labels=causes)
    df["p10"].loc[df["p10"].isnull()] = "iné"
    df = df.loc[df["date"].dt.year != 2021]
    df["tmp_date"] = df["date"].dt.month

    tmp = (
        df[["region", "p10", "tmp_date", "p1"]]
        .groupby(["region", "tmp_date", "p10"])
        .agg({"p1": "count"})
        .reset_index()
    )

    figure, axes = plt.subplots(
        nrows=2, ncols=2, constrained_layout=True, figsize=(10, 10)
    )

    counter = 0

    for el in regions:
        tmp2 = pd.pivot(
            tmp.loc[tmp["region"] == el], columns="p10", values="p1", index="tmp_date"
        )

        ax = tmp2.plot(
            ax=axes[int(counter - 2 >= 0)][counter % 2],
            kind="bar",
            color=colors,
            title="KRAJ {}".format(el),
        )
        ax.set_facecolor("lavender")
        ax.set_title(el)
        ax.get_legend().remove()
        handles, labels = ax.get_legend_handles_labels()
        lgd = figure.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))
        if counter - 2 < 0:
            ax.axes.xaxis.set_visible(False)

        if counter == 0 or counter == 2:
            ax.set_ylabel("Pocet nehod")

        if counter == 2 or counter == 3:
            ax.set_xlabel("mesiac")

        counter += 1

    if fig_location is not None:
        dir = fig_location.rsplit("/", 1)
        if len(dir) > 1:
            make_folder_if_not_exist(dir[0])
        plt.savefig(fig_location, bbox_extra_artists=(lgd,), bbox_inches="tight")

    if show_figure:
        plt.show()
        


# Ukol 4: Povětrnostní podmínky
def plot_conditions(
    df: pd.DataFrame, fig_location: str = None, show_figure: bool = False
):
    sns.set_palette("dark")
    figure, axes = plt.subplots(
        nrows=2, ncols=2, constrained_layout=True, figsize=(10, 10)
    )

    if df is None:
        df = get_dataframe()

    df = df.loc[((df["region"].isin(regions)) & (df["p18"] != 0))].copy()
    df["p18"] = pd.cut(df["p18"], [0, 1, 2, 3, 4, 5, 6, 7], labels=cases_weather)
    tmp = pd.pivot_table(
        df[["date", "region", "p18"]],
        index=df[["region", "date"]],
        columns=df["p18"],
        aggfunc="count",
    )
    counter = 0

    for el in regions:
        a = tmp.loc[el, "p18"].resample("M").sum().stack().reset_index()
        ax = sns.lineplot(
            ax=axes[int(counter - 2 >= 0)][counter % 2],
            data=a,
            hue="p18",
            x="date",
            y=0,
        )

        ax.set_facecolor("lavender")
        handles, labels = ax.get_legend_handles_labels()
        lgd = figure.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
            title="Podminky",
        )
        ax.set_xlim([datetime.date(2016, 1, 1), datetime.date(2021, 1, 1)])
        ax.get_legend().remove()
        ax.set_title("KRAJ {}".format(el))
        ax.xaxis.label.set_visible(False)

        if counter == 0 or counter == 2:
            ax.set_ylabel("Pocet nehod")
        else:
            ax.axes.yaxis.label.set_visible(False)

        counter += 1

    if fig_location is not None:
        dir = fig_location.rsplit("/", 1)
        if len(dir) > 1:
            make_folder_if_not_exist(dir[0])
        plt.savefig(fig_location, bbox_extra_artists=(lgd,)
        , bbox_inches="tight")

    if show_figure:
        plt.show()
        


if __name__ == "__main__":
    # zde je ukazka pouziti, tuto cast muzete modifikovat podle libosti
    # skript nebude pri testovani pousten primo, ale budou volany konkreni ¨
    # funkce.
    df = get_dataframe(
        "accidents.pkl",verbose=True
    )  # tento soubor si stahnete sami, při testování pro hodnocení bude existovat
    plot_roadtype(df, fig_location="01_roadtype.pdf", show_figure=True)
    plot_animals(df, "02_animals.pdf", True)
    plot_conditions(df, "03_conditions.pdf", True)


