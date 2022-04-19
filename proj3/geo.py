#!/usr/bin/python3.8
# coding=utf-8
import pandas as pd
import geopandas
import contextily as ctx
import matplotlib.pyplot as plt
import contextily
import sklearn.cluster
import numpy as np
import numpy as np
# muzete pridat vlastni knihovny


## this function finds max and min of specific column in given dataframe
def get_max_min(df,column):
    max = df[column].max()
    min = df[column].min()

    return max,min

years_to_plot = [2018,2019,2020]
to_plot = {0:[0,0],1:[1,0],2:[2,0],3:[0,1],4:[1,1],5:[2,1]}
high_ways = 0
roads = 1

def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    df = df.loc[((~df["e"].isnull()) & (~df["d"].isnull()))].copy()
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df[["d"]], df[["e"]]),crs="EPSG:5514")
    return gdf

def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    
    gdf["date"] = gdf["p2a"].astype("datetime64")

    figure, axes = plt.subplots(
        nrows=3, ncols=2, figsize=(10, 15)
    )

    ## for spaces between graphs
    figure.tight_layout(pad=3.0)
    
    counter = 0

    ## FINDING MIN and max because we want graphs of same lenghth
    tmp = gdf.loc[((gdf["p36"].isin([0,1])) & (gdf["region"] == "JHM"))]
    maxx,minx = get_max_min(tmp,"d")
    maxy,miny = get_max_min(tmp,"e")

    # turning off axis 
    [axi.set_axis_off() for axi in axes.ravel()]

    # loop throught each year
    for year in years_to_plot: 

        # for highways
        ax = axes[to_plot[counter][0]][to_plot[counter][1]]
        ax.set_xlim(minx,maxx)
        ax.set_ylim(miny,maxy)       
        gdf.loc[(gdf["p36"] == high_ways) & (gdf["region"] == "JHM") & (gdf["date"].dt.year == year)].plot(ax=ax,markersize=1,color="red")
        
        ctx.add_basemap(ax=ax, crs=gdf.crs.to_string(), source=ctx.providers.Stamen.TonerLite, alpha=0.9)

        ax.set_title("JHM kraj: dialnica({})".format(year))

        # for roads
        ax = axes[to_plot[counter + 3][0]][to_plot[counter + 3][1]]
        ax.set_xlim(minx,maxx)
        ax.set_ylim(miny,maxy) 
        
        gdf.loc[(gdf["p36"] == roads) & (gdf["region"] == "JHM") & (gdf["date"].dt.year == year)].plot(ax=ax,markersize=1,color="green")
        ctx.add_basemap(ax=ax, crs=gdf.crs.to_string(), source=ctx.providers.Stamen.TonerLite)
        ax.set_title("JHM kraj: cesta pr vej triedy({})".format(year))
        
        counter += 1

    if fig_location is not None:
        plt.savefig(fig_location)

    if show_figure:
        plt.show()
    

def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):

    gdf_1 = gdf.loc[(gdf["region"] == "JHM") & (gdf["p36"] == 1)].copy()
    coords = np.dstack([gdf_1.d,gdf_1.e]).reshape(-1,2)

    # I tried 3 cluster method first was
    # sklearn.cluster.MiniBatchKMeans(n_clusters=24) -> but here i never had (in range 10 - 30 n_clusters) that much accidents in two places
    # sklearn.cluster.KMeans(n_clusters=24) -> but here alsp  i never had (in range 10 - 30 n_clusters) that much accidents in two places
    # this was fine

    model = sklearn.cluster.AgglomerativeClustering(n_clusters=23)

    db = model.fit(coords)

    gdf_1["cluster"] = db.labels_
    gdf_1["size"] = gdf_1.groupby('cluster')['cluster'].transform('size')
   
    ax = gdf_1.plot(column="size",legend=True,legend_kwds={'orientation':'horizontal','fraction':0.05,'pad':0.04},markersize=1,figsize=(10,10))

    ax.set_axis_off()
    ctx.add_basemap(ax=ax, crs=gdf.crs.to_string(), source=ctx.providers.Stamen.TonerLite)

    if fig_location is not None:
        plt.savefig(fig_location)

    if show_figure:
        plt.show()

if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    gdf = make_geo(pd.read_pickle("accidents.pkl"))
    plot_geo(gdf, "geo1.pdf", True)
    plot_cluster(gdf, "geo2.pdf", True)
