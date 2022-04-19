#!/usr/bin/python3.8
# coding=utf-8
import pandas as pd
import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

year_of_made = ["nezadane", "2001-2005","2006-2010","2011-2015","2016-2021","<1990","1990-1995","1996-1999"]
sorted_years_of_made = ["<1990","1990-1995","1996-1999","2001-2005","2006-2010","2011-2015","2016-2021","nezadane"]

causes = ["nezavinena ridicem", "nepřiměřená rychlost jízdy","nesprávné předjíždění",
        "nedání přednosti v jízdě","nesprávný způsob jízdy",
        "technická závada vozidla"]

def group_sum(x):
    return x.sum()

def get_dataframe() -> pd.DataFrame:
    df = pd.read_pickle("accidents.pkl")

    df = pd.DataFrame.from_dict(df)

    # changing types of some object types to category
    column_to_change = ["g", "h", "k", "p", "t", "s", "o", "j", "i"]
    df[column_to_change] = df[column_to_change].apply(
        lambda x: x.astype("category")
    )

    # rename and change type to date
    df["p2a"] = df["p2a"].astype("datetime64")
    df.rename(columns={"p2a": "date"}, inplace=True)

    # grouping years of made cars in intervals
    df["p47"] = pd.cut(df["p47"], [-2,-1,5,10,15,21,90,95,99],labels=year_of_made)

    return df

def plot_percentage_mortage(df, fig_location: str = None,
                 show_figure: bool = False):
    
    # counting sum of all deads in crash and counting all crashes
    df = df.groupby(["p47"]).agg({"p13a": "sum","p1":"count"}).copy()

    # counting percentage of mortality in crash
    df["new"] = df["p13a"]/df["p1"]
    mortality_by_year = df["new"] * 100

    # sorted from oldest to newest
    mortality_by_year = mortality_by_year.reindex(sorted_years_of_made)

    ax = mortality_by_year.plot.bar()
    ax.set_xlabel("rok vyroby")
    ax.set_ylabel("miera umrtnosti voci zrazkam daneho rocnika")
   

    if fig_location is not None:
        plt.savefig(fig_location)

    if show_figure:
        plt.show()

def plot_causes_by_year(df, fig_location: str = None,
                 show_figure: bool = False):

    # sum of all crashes by year and cause
    df = df.groupby(["p47","p12"]).agg({"p1":"count"}).reset_index().copy()

    df["p12"] = pd.cut(df["p12"],[-1,200,300,400,500,600,700],labels=causes)
    df = df[df["p47"] != "nezadane"] 


    sum = df.groupby(["p47"]).agg({"p1":"sum"}).reset_index()["p1"]
    zip_iterator = zip(year_of_made, sum) 
    dictionary = dict(zip_iterator) 

    for el in dictionary:
        df.loc[df["p47"] == el, "p1"] = df[df["p47"] == el]["p1"] / dictionary[el] * 100
    
    df = pd.pivot_table(df,index=["p47","p12"]).reindex(sorted_years_of_made,level=0).reset_index()


    figure, axes = plt.subplots(
            nrows=2, ncols=3, constrained_layout=True, figsize=(10, 10)
        )

    counter = 0
    for cause in causes:
        
        tmp = df.loc[df["p12"] == cause]
        ax = sns.barplot(
                ax=axes[int(counter - 3 >= 0)][counter % 3],
                x="p47",
                y="p1",
                data=tmp   
            )
        
        if counter == 0 or counter == 3:
            ax.set_ylabel("percentualne zastupenie nehod")
        else:
            ax.axes.yaxis.label.set_visible(False)

        if counter >= 3:
            ax.set_xlabel("rok vyroby auta")
            ax.tick_params(axis='x', rotation=90) 
        else:
            ax.axes.xaxis.label.set_visible(False)
            ax.tick_params(axis='x', labelbottom=False) 

        ax.set_facecolor("lavender")
        ax.set_title(cause)
        counter += 1

    ax.tick_params(axis='x', rotation=90) 
    
    if fig_location is not None:
        plt.savefig(fig_location)

    if show_figure:
        plt.show()

def print_table(df,get_dataframe=False,prints=False):

    # sort out accidents where noone was drugged or drunked
    df = df.loc[((df["p11"] != 2) | (df["p11"] != 0))]

    # we want just months
    df["date"] = df["date"].dt.strftime('%m')

    # sum of acidents in each region each month 
    df = df.groupby(["region","date"]).agg({"p11":"sum"}).reset_index()

    # counting max value
    df["max"] = df.groupby(['region'], sort=False)["p11"].transform(max)

    # counting sum of all in that region
    df["sum"] = df.groupby(['region'], sort=False)["p11"].transform(group_sum)

    df = df.rename(columns={"date":"month"})
    

    # finding max value in months
    a = df.loc[df["p11"] == df["max"]].copy()
    
    a["percentage"] = round(a["max"] / a["sum"] * 100, 2)

    if prints:
        a.to_csv("table.csv")
        print(a.drop(columns=["p11"]).reset_index(drop=True))
    if get_dataframe:
        return a
    
def print_facts(df):

    tmp = df[["date","p13a","p13b","p13c","p12"]]

    # count how many injuries of persons without driver
    tmp = tmp.loc[(tmp["p12"] > 200) &  (tmp["p12"] < 600) ]
    tmp = tmp.rename(columns = {'p13a': 'smrt','p13b':'s nasledkami','p13c' : 'lahke poranenie'}, inplace = False)
    tmp["date"] = tmp["date"].dt.year

    tmp4 = tmp[['smrt',"s nasledkami"]].sum()

    all = tmp.count().max()
    how_many = tmp.loc[(tmp['smrt'] > 1) | (tmp['s nasledkami'] > 1) | (tmp['lahke poranenie'] > 1)].count().max() 

    print("in each accident caused by driver: ", round(all/how_many,2), " was injured someone but driver" )

    # months where drivers was drunked and caused the most accidents

    tmp2 = print_table(df,get_dataframe=True)
    tmp2 = tmp2["month"].drop_duplicates().to_numpy() 
    print("in these months were the most accidents made by drunk or drug drivers",tmp2)

    # region where the most acidents were caused by drunk or drugged drivers in scale with total crashes when police was tested them

    t = df
    t = t.loc[t["p11"] != 0].copy()
    t.loc[t["p11"] != 2,"p11"] = 1

    t = t.groupby(["region","p11"]).agg({"p1":"count"}).reset_index()

    t = pd.pivot_table(t, columns="p11", values="p1", index="region",aggfunc="sum")
    t.reset_index()

    t["percentage"] = round(t[1] / (t[1] + t[2]) * 100,2)

    print("najviac vodicov boli pod vplivom omamnej latky (aj alkoholu) ked zistovala policia v kraji :", 
    t.loc[t["percentage"] == t["percentage"].max()].reset_index().region.values[0] ," kraji a to az ", t["percentage"].max(), "%")


if __name__ == "__main__":
    df = get_dataframe()
    plot_percentage_mortage(df,fig_location="fig_mortage_by_year_of_made")
    plot_causes_by_year(df,fig_location="fig_causes_by_year_of_made")
    print_table(df,prints=True)
    print_facts(df)