"""
Functions are useful untilities for data processing in the NN

Notes
-----
    Author : Zachary Labe
    Date   : 8 July 2020

Usage
-----
    [1] readFiles(variq,dataset)
    [2] getRegion(data,lat1,lon1,lat_bounds,lon_bounds)

"""
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import cmocean
import cartopy.crs as ccrs
import xarray as xr
import matplotlib as mpl

from ..visuals.general import *

def plot_XAImaps(
        data: xr.DataArray,
        xai_methods: Dict,
        **params):

    ticks, ticklabs = create_tickscheme(data, **params)
    labels = create_labels(data, **params)
    ls = params['plot']['labelsize']
    fs = params['plot']['fontsize']
    types = params['plot']['types']
    colorf = params['plot']['colorf']
    limits = create_limits(data, **params)
    cmaps = create_cmap(data, **params)

    ### Plot variable data for trends
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
    plt.rc('savefig', facecolor='white')
    plt.rc('axes', edgecolor='darkgrey')
    plt.rc('xtick', color='black')
    plt.rc('ytick', color='black')
    plt.rc('axes', labelcolor='darkgrey')
    plt.rc('axes', facecolor='white')

    yearperiod = data.periods.values
    xai_methods = data.models.values

    # hg_ratio = params['plot']['hg_ratio']


    fig1, axes = plt.subplots(nrows=len(yearperiod), ncols=len(xai_methods), figsize=params['plot']['figsize'],
                              subplot_kw={'projection': ccrs.Robinson(central_longitude=0)},)
                              # gridspec_kw={"height_ratios": hg_ratio,
                              #              "hspace": 0.1})

    plt_kw = {}
    plt_kw['trafo'] = ccrs.PlateCarree()



    c = 0

    for mod in range(len(data.models.values)):

        dat = data.loc[data.models.values[mod]]

        for yp in range(len(yearperiod)):

            if len(yearperiod)>1:
                ax = axes[yp,mod]
            else:
                ax = axes[mod]

            ax.coastlines(resolution='110m', color='dimgrey', linewidth=0.35)
            ax.set_global()

            if any(params['plot']['region']):
                ax.set_extent(params['plot']['region'], crs=plt_kw['trafo'])
                ax.gridlines(color='black', linewidth=0.25)

            plt_kw['cmap'] = cmaps[mod]
            plt_kw['min'] = limits[0, mod]
            plt_kw['max'] = limits[1, mod]

            vp = plot_on_axis(ax, dat.loc[yearperiod[yp]], types,**plt_kw)
            # if mod > 0:
            #  ax = set_frame(ax, mod, colorf, 5)

            # if params['add_raw'] and mod == 0:
            #     cbar_ax = fig1.add_axes(params['plot']['raxis'])
            #     cbar1 = fig1.colorbar(vp, cax=cbar_ax, orientation='horizontal')
            #     cbar1.ax.tick_params(axis='x', size=.02, labelsize=ls)
            #     cbar1.outline.set_edgecolor('darkgrey')
            #     cbar1.set_ticks(ticks[mod])
            #     cbar1.set_ticklabels(ticklabs[mod])
            #     cbar1.set_label(labels[0], fontsize=fs, color='black', labelpad=1.4)
            #
            # elif 'LRPz' in xai_methods[mod]:
            #     cbar_ax = fig1.add_axes(params['plot']['Eaxis'])
            #     cbar1 = fig1.colorbar(vp, cax=cbar_ax, orientation='horizontal')
            #     cbar1.ax.tick_params(axis='x', size=.02, labelsize=ls)
            #     cbar1.outline.set_edgecolor('darkgrey')
            #     cbar1.set_ticks(ticks[mod])
            #     cbar1.set_ticklabels(ticklabs[mod])
            #     cbar1.set_label(labels[1], fontsize=fs, color='black', labelpad=1.4)


            if yp == 0:
            # if (c % len(xai_methods)) == 0:
                ax.annotate(r'\textbf{%s}' % data.models.values[mod], xy=(0, 0), xytext=(0.5, 1.22),
                            textcoords='axes fraction', color='darkgrey', fontsize=fs,
                            rotation=0, ha='center', va='center')
            # if (c % len(yearperiod)) == 0:
            if mod == 0:
                ax.annotate(r'\textbf{%s}' % yearperiod[yp], xy=(0, 0), xytext=(-0.1, 0.5),
                            textcoords='axes fraction', color='darkgrey', fontsize=fs,
                            rotation=90, ha='center', va='center')

            c += 1


    fig1.subplots_adjust(top=0.85, wspace=0.01, hspace=0, bottom=0.14)
    fig1.savefig(params['plot']['dir'] + params['plot']['figname'], dpi=600, bbox_inches='tight')

    plt.close(fig1)

    return

def plot_HPmaps(
        data,
        **params):


    fs = params['plot']['fontsize']
    types = params['plot']['types']

    # limits = params['plot']['limits']#create_limits(data, **params)
    cmaps = params['plot']['cmaps']#create_cmap(data, **params)
    if len(data.models.values) != len(params['plot']['cmaps']):
        cmaps = create_cmap(data, **params)

    ### Plot variable data for trends
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
    plt.rc('savefig', facecolor='white')
    plt.rc('axes', edgecolor='darkgrey')
    plt.rc('xtick', color='black')
    plt.rc('ytick', color='black')
    plt.rc('axes', labelcolor='darkgrey')
    plt.rc('axes', facecolor='white')

    yearperiod = data.periods.values
    methods = data.models.values



    fig1, axes = plt.subplots(nrows=len(yearperiod), ncols=len(methods), figsize=params['plot']['figsize'],
                              subplot_kw={'projection': ccrs.Robinson(central_longitude=0)},)


    plt_kw = {}
    plt_kw['trafo'] = ccrs.PlateCarree()



    c = 0

    for mod in range(len(data.models.values)):

        dat = data.loc[data.models.values[mod]]

        for yp in range(len(yearperiod)):
            if len(yearperiod)>1:
                ax = axes[yp,mod]
            else:
                ax = axes[mod]

            ax.coastlines(resolution='110m', color='dimgrey', linewidth=0.35)
            ax.set_global()

            plt_kw['cmap'] = cmaps[mod]
            # plt_kw['min'] = limits[0]
            # plt_kw['max'] =  limits[1]
            plt_kw['min'] = dat.loc[yearperiod[yp]].min().values#limits[0]
            plt_kw['max'] = dat.loc[yearperiod[yp]].max().values#limits[1]

            vp = plot_on_axis(ax, dat.loc[yearperiod[yp]], types,**plt_kw)


            if yp == 0:

                ax.annotate(r'\textbf{%s}' % data.models.values[mod], xy=(0, 0), xytext=(0.5, 1.22),
                            textcoords='axes fraction', color='darkgrey', fontsize=fs,
                            rotation=0, ha='center', va='center')

            if mod == 0:
                ax.annotate(r'\textbf{%s}' % yearperiod[yp], xy=(0, 0), xytext=(-0.1, 0.5),
                            textcoords='axes fraction', color='darkgrey', fontsize=fs,
                            rotation=90, ha='center', va='center')

            c += 1


    fig1.subplots_adjust(top=0.85, wspace=0.01, hspace=0, bottom=0.14)
    fig1.savefig(params['plot']['dir'] + params['plot']['figname'], dpi=600, bbox_inches='tight')

    plt.close(fig1)

    return

def plot_MapsCB(
        data,
        **params):


    fs = params['plot']['fontsize']
    types = params['plot']['types']

    # limits = params['plot']['limits']#create_limits(data, **params)
    cmaps = params['plot']['cmaps']#create_cmap(data, **params)
    if len(data.models.values) != len(params['plot']['cmaps']):
        cmaps = create_cmap(data, **params)

    ### Plot variable data for trends
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
    plt.rc('savefig', facecolor='white')
    plt.rc('axes', edgecolor='darkgrey')
    plt.rc('xtick', color='black')
    plt.rc('ytick', color='black')
    plt.rc('axes', labelcolor='darkgrey')
    plt.rc('axes', facecolor='white')

    yearperiod = data.periods.values
    methods = data.models.values



    fig1, axes = plt.subplots(nrows=len(yearperiod), ncols=len(methods), figsize=params['plot']['figsize'],
                              subplot_kw={'projection': ccrs.Robinson(central_longitude=0)},)


    plt_kw = {}
    plt_kw['trafo'] = ccrs.PlateCarree()



    c = 0

    for mod in range(len(data.models.values)):

        dat = data.loc[data.models.values[mod]]

        for yp in range(len(yearperiod)):
            if len(yearperiod)>1:
                ax = axes[yp,mod]
            else:
                ax = axes[mod]

            ax.coastlines(resolution='110m', color='dimgrey', linewidth=0.35)
            ax.set_global()

            plt_kw['cmap'] = cmaps[mod]
            # plt_kw['min'] = limits[0]
            # plt_kw['max'] =  limits[1]
            # plt_kw['min'] = -1
            # plt_kw['max'] = 1.
            plt_kw['min'] = dat.loc[yearperiod[yp]].min().values#limits[0]
            plt_kw['max'] = dat.loc[yearperiod[yp]].max().values#limits[1]

            vp = plot_with_map(ax, dat.loc[yearperiod[yp]], types,**plt_kw)


            if yp == 0:

                ax.annotate(r'\textbf{%s}' % data.models.values[mod], xy=(0, 0), xytext=(0.5, 1.22),
                            textcoords='axes fraction', color='darkgrey', fontsize=fs,
                            rotation=0, ha='center', va='center')

            if mod == 0:
                ax.annotate(r'\textbf{%s}' % yearperiod[yp], xy=(0, 0), xytext=(-0.1, 0.5),
                            textcoords='axes fraction', color='darkgrey', fontsize=fs,
                            rotation=90, ha='center', va='center')

            c += 1


    fig1.subplots_adjust(top=0.85, wspace=0.01, hspace=0, bottom=0.14)
    fig1.savefig(params['plot']['dir'] + params['plot']['figname'], dpi=600, bbox_inches='tight')

    plt.close(fig1)

    return

def plot_PaperMaps(
        data: xr.DataArray,
        **params):

    ticks, ticklabs = create_tickscheme(data, **params)
    labels = create_labels(data, **params)
    ls = params['plot']['labelsize']
    fs = params['plot']['fontsize']
    types = params['plot']['types']
    colorf = params['plot']['colorf']
    limits = create_limits(data, **params)
    cmaps = create_cmap(data, **params)

    ### Plot variable data for trends
    # mpl.rcParamsDefault['text.usetex'] = True
    # mpl.rcParamsDefault['text.latex.preamble'] = [r'\usepackage{amsmath}'] 
    mpl.rcParamsDefault['font.weight'] = 'bold'
    # mpl.rcParams.update(plt.rcParamsDefault)
    # plt.rc('text', usetex=plt.rcParamsDefault['text.usetex'])
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
    plt.rc('savefig', facecolor='white')
    plt.rc('axes', edgecolor='darkgrey')
    plt.rc('xtick', color='black')
    plt.rc('ytick', color='black')
    plt.rc('axes', labelcolor='darkgrey')
    plt.rc('axes', facecolor='white')

    yearperiod = data.periods.values
    xai_methods = data.models.values

    # hg_ratio = params['plot']['hg_ratio']


    fig1, axes = plt.subplots(nrows=len(xai_methods), ncols=len(yearperiod), figsize=params['plot']['figsize'],
                              subplot_kw={'projection': ccrs.Robinson(central_longitude=0)},)
                              # gridspec_kw={"height_ratios": hg_ratio,
                              #              "hspace": 0.1})

    plt_kw = {}
    plt_kw['trafo'] = ccrs.PlateCarree()



    c = 0

    for mod in range(len(data.models.values)):

        dat = data.loc[data.models.values[mod]]

        for yp in range(len(yearperiod)):

            if len(yearperiod)>1:
                ax = axes[mod,yp]
            else:
                ax = axes[mod]

            ax.coastlines(resolution='110m', color='dimgrey', linewidth=0.4)
            ax.set_global()

            plt_kw['cmap'] = cmaps[mod]
            if mod == 0:
                plt_kw['min'] = dat.loc[yearperiod[yp]].min().values  # limits[0]
                plt_kw['max'] = dat.loc[yearperiod[yp]].max().values  # limits[1]
            else:
                plt_kw['min'] = limits[0, mod]
                plt_kw['max'] = limits[1, mod]

            vp = plot_on_axis(ax, dat.loc[yearperiod[yp]], types,**plt_kw)

            if mod == 0:
                # ax.annotate(r'$\mathbf{%s}$' % yearperiod[yp], xy=(0, 0), xytext=(0.5, 1.22),
                #             textcoords='axes fraction', color='darkgrey', fontsize=fs,
                #             rotation=0, ha='center', va='center')
                ax.annotate(r'%s' % yearperiod[yp], xy=(0, 0), xytext=(0.5, 1.22),
                            textcoords='axes fraction', color='darkgrey', fontsize=fs,
                            rotation=0, ha='center', va='center')

            if yp == 0:
                # ax.annotate(r'$\mathbf{%s}$' % data.models.values[mod], xy=(0, 0), xytext=(-0.1, 0.5),
                #             textcoords='axes fraction', color='darkgrey', fontsize=fs,
                #             rotation=90, ha='center', va='center')
                ax.annotate(r'%s' % data.models.values[mod], xy=(0, 0), xytext=(-0.1, 0.5),
                            textcoords='axes fraction', color='darkgrey', fontsize=fs,
                            rotation=90, ha='center', va='center')

            if any(params['plot']['region']):
                ax.set_extent(params['plot']['region'], crs=plt_kw['trafo'])
                ax.gridlines()

            ax.coastlines(resolution='110m', color='dimgrey', linewidth=0.4)
            ax.set_global()
            
            c += 1


    fig1.subplots_adjust(top=0.85, wspace=0.01, hspace=0, bottom=0.14)
    fig1.savefig(params['plot']['dir'] + params['plot']['figname'], dpi=600, bbox_inches='tight')

    plt.close(fig1)

    return