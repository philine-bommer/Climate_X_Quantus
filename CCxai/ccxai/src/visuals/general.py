"""
Functions are useful untilities for data processing in the NN

Notes
-----
    Author : Philine Bommer
    Date   : 11 Feburary 2022

Usage
-----
    [1] readFiles(variq,dataset)
    [2] getRegion(data,lat1,lon1,lat_bounds,lon_bounds)

"""
from typing import Dict, List
import sys
import matplotlib.pyplot as plt
import numpy as np
import cmocean
import cartopy.crs as ccrs
import xarray as xr
import matplotlib.colors as colors
import pdb
import matplotlib.colors as mplc

def plot_on_axis(ax, data, types,**plot_kwargs):


    if types == 'continuous':
        obj = data.plot(ax=ax, transform=plot_kwargs['trafo'], cmap=plot_kwargs['cmap'], vmin=plot_kwargs['min'], vmax=plot_kwargs['max'],
                                add_colorbar=False, add_labels=False)
    else:
        obj = data.plot(ax=ax, transform=plot_kwargs['trafo'], cmap=plot_kwargs['cmap'],
                                 vmin=plot_kwargs['min'], vmax=plot_kwargs['max'], add_colorbar=False, add_labels=False)


    return obj
def plot_with_map(ax, data, types,**plot_kwargs):


    if types == 'continuous':
        obj = data.plot(ax=ax, transform=plot_kwargs['trafo'], cmap=plot_kwargs['cmap'], vmin=plot_kwargs['min'], vmax=plot_kwargs['max'],
                                add_colorbar=True, add_labels=False)
    else:
        obj = data.plot(ax=ax, transform=plot_kwargs['trafo'], cmap=plot_kwargs['cmap'],
                                 vmin=plot_kwargs['min'], vmax=plot_kwargs['max'], add_colorbar=True, add_labels=False)


    return obj

def set_frame(axis, idx, colors ,wd = 5):

    try:
        axis.xaxis.set_visible([])
        axis.yaxis.set_visible([])
        axis.spines["top"].set_color(colors[idx - 1])
        axis.spines["bottom"].set_color(colors[idx - 1])
        axis.spines["left"].set_color(colors[idx - 1])
        axis.spines["right"].set_color(colors[idx - 1])
        axis.spines["top"].set_linewidth(wd)
        axis.spines["bottom"].set_linewidth(wd)
        axis.spines["left"].set_linewidth(wd)
        axis.spines["right"].set_linewidth(wd)
        res = 1
    except:
        res = 0
        pdb.set_trace()
        print("Configuration mishap:", sys.exc_info()[0])

    return axis

def create_labels(data, **params):
    ''' Creates label array for the colorbar according to depicted data'''
    labels = []
    for mod in range(len(data.models.values)):
        if params['plot']['add_raw'] and mod == len(data.models.values) - 1:
            labels.append(r'\textbf{standized. T}')
        else:
            labels.append(params['plot']['label'])


    return labels


def create_limits(data, **params):
    ''' Creates limits array for the plot according to depicted data'''
    limits = np.zeros((2,len(data.models.values)))


    for mod in range(len(data.models.values)):
        expl = data[{'models':mod}]
        xmax = np.max(np.array([np.abs(expl.min().values), expl.max().values]))
        xmin = -xmax
        limits[0, mod] = xmin
        limits[1, mod] = xmax

    return limits


def create_cmap(data, **params):
    ''' Creates cmap name array for the colorbar according to depicted data'''
    cmaps = []
    if params['plot']['types'] == 'continuous':
        for mod in range(len(data.models.values)):
            vals = data.loc[data.models.values[mod]]
            if params['plot']['add_raw'] and mod == 0:
                cmaps.append("coolwarm")
            else:
                cmaps.append("seismic")
                # if vals.min().values < 0:
                #     cmaps.append("seismic")
                # else:
                #     cmaps.append("Reds")


    return cmaps



def create_tickscheme(data, **params):
    ''' Creates level array for the colorbar according to depicted data'''

    ticks = []
    labs = []

    for mod in range(len(data.models.values)):
        vals = data.loc[data.models.values[mod]]
        if np.abs(vals).max().values > 1.1:
            absmax = np.abs(vals).max().values
            ticks.append(np.asarray(np.linspace(-absmax, absmax, 5)))
            labs.append(np.around(np.asarray(np.linspace(-absmax, absmax, 5)), 1).astype(str))
        else:
            if vals.min().values <0:
                ticks.append(np.asarray(np.linspace(-1., 1., 6)))
                labs.append(np.around(ticks[mod], 1).astype(str))
            else:
                ticks.append(np.asarray(np.linspace(0., 1., 5)))
                labs.append(np.around(ticks[mod], 1).astype(str))




    return ticks, labs

def climits(data, **params):
    '''
    Create limits for the cbar
    :param data:
    :param params:
    :return:
    '''

    limits = np.zeros((len(data.models.values),2))
    for mod in range(len(data.models.values)):
        vals = data.loc[data.models.values[mod]]

        if np.min(vals.values) < 0 and np.max(vals.values) >= -1:
            limits[mod, 0] = -1
            limits[mod, 1] = 1
        elif np.min(vals.values) < -1:
            absmax = np.max(vals.values)
            limits[mod, 0] = -absmax
            limits[mod, 1] = absmax
        else:
            limits[mod, 0] = 0
            limits[mod, 1] = 1
    return limits