#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from PIL.Image import LINEAR
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import Colormap, LogNorm
import os.path
import argparse
import sys
# povolene jsou pouze zakladni knihovny (os, sys) a knihovny numpy, matplotlib a argparse

from download import DataDownloader

# this function sets to ax graph y names of axis and x names of axis
def set_ticks(ax, x_axis, y_axis):
    ax.set_xticks(range(len(x_axis)))
    ax.set_xticklabels(x_axis)

    ax.set_yticks(range(len(y_axis)))
    ax.set_yticklabels(y_axis)

    return ax

def make_folder_if_not_exist(folder):
        folder_to_make = os.getcwd() + "/" + folder
    
        if not os.path.exists(folder_to_make):
            os.makedirs(folder_to_make)

def plot_stat(data_source,
              fig_location=None,
              show_figure=False):

    #getting data
    data_source = DataDownloader().get_dict()
    
    #substitute string of region with numbers
    integers_for_region = np.array(np.unique(data_source["region"],return_inverse=True)[1].tolist())

    # comlums p24 -> substitute all items with value 0 with 6
    # because 0 finction np.astype return size of matrix but max number is 
    # size and zero is not include here and also in school assignment this
    # column was as index 6 in graph 
    data_source["p24"][data_source["p24"] == 0] = 6

    # also because of 0 and function np.zeros 
    integers_for_region += 1

    # creating matrix with shape [region, type] for each element (index)
    tmp = np.column_stack((integers_for_region,data_source["p24"]))
    
    # create matrix with shape of max numbers in index 0 column
    # and index i (row) 
    result = np.zeros(tmp.max(0).astype(int), np.int32)

    # fullfiling matrix with count of types for each region
    np.add.at(result, tuple(tmp.astype(int).T -1), 1)

    # for second graph we need transpose of first matrix
    # also need to retype it to float because percentage 
    # should be precisely
    result2 = np.array(result,dtype=float).T
    
    # count percentage of cause for each region (1 cause is 100% and calculated how many % has each region) 
    for i in range(0, len(result2)):
        sum_of_list = sum(result2[i])
        result2[i] = result2[i] / float(sum_of_list) * 100

    # for white tab in graph
    result2[result2 == 0] = np.NaN

    fig, axs = plt.subplots(2,constrained_layout=True)
    

    x_axis = np.array(np.unique(data_source["region"]))
    y_axis = np.array(["Prerusovana zlta","Semafor mimo prevadky",
                       "Dopravne znacky","Prenosne dopravne znacky",
                       "nevyznacene","Ziadna uprava"])
   
    axs[0] = set_ticks(axs[0],x_axis,y_axis)
    axs[0].set_title("relativne voci pricine")

    graph = fig.colorbar(axs[0].imshow(result2),ax=axs[0])
    graph.ax.set_ylabel('podiel nehod pre danu pricinu [%]', rotation=90)
  
    axs[1] = set_ticks(axs[1],x_axis,y_axis)
    axs[1].set_title("absolutne")

    graph = axs[1].imshow(result.T,norm=LogNorm(vmin=1,vmax=100000),interpolation='nearest')

    colorbar = fig.colorbar(graph)
    colorbar.ax.set_ylabel('podiel nehod', rotation=90)

    if fig_location != None:
        dir = fig_location.rsplit("/", 1)
        if len(dir) > 1:
            make_folder_if_not_exist(dir[0])
        plt.savefig(fig_location)
    
    if show_figure:
        plt.show()

data = dict()

parser = argparse.ArgumentParser()

parser.add_argument('--fig_location', type=str)
parser.add_argument('--show_figure', action='store_true')

args = vars(parser.parse_args())

plot_stat(data,fig_location=args["fig_location"],show_figure=args["show_figure"])

if __name__ == '__main__':
    plot_stat(data,fig_location="result.pdf")

