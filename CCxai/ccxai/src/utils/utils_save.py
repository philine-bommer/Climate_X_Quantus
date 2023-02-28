"""
Utils regarding data formatting and saving

Author    : Philine Bommer
Date      : 24 January 2022
"""
### Import packages
from typing import Tuple, Optional, Any, List,Dict
import palettable.cubehelix as cm
import matplotlib as mpl
import numpy as np
import xarray as xr
import yaml
import os
import sys
import pdb
import csv
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def savedatasegments(data: xr.DataArray,
                     ** params):

    """
    Save indivual segments of any xarray dataarray
    :param data: data to save as xarray datarray (necessary dimensions: models, time)
    :param params: dict with data description and directories {dtype: type of values in dataarray
            dir: data directory}
    :return: boolean (true in case of sucessful storage)
    """

    directorydata = params['save']['dir']
    model = data.models.values[len(data.models.values)-1]
    time = data.times.values
    dtype = params['save']['dtype']
    figname = '%s_%s_%s-%s_iso.nc' % (dtype, model,  time[0], time[len(time)-1])
    try:
        data.to_netcdf(directorydata + figname)
        saves = True
        print('Saving %s data snippet was successful' %dtype)
    except:
        saves = False

    return saves

def yrs_inDataset(data, indx, dim):
    """
    Function: constructing dataarray in plot format for plot_xrMaps
    :param data: xr map (4 dimensional)
    :param indx: list of chosen indicies (int arrays)
    :param dim: string = name coordinate
    :return: DataArray
    """


    # vals = data.values
    lats = data['lat'].values
    lons = data['lon'].values
    dats = data[{dim:indx}]


    # catArray = xr.DataArray(data=dats, dims=['models', 'time', 'lat', 'lon'],
    #                         coords=dict(models=modelsname,
    #                                     time=yrs, lon=("lon", lons),
    #                                     lat=("lat", lats)),
    #                         attrs=dict(description='Diff maps for random sampling of each year',
    #                                    units='unitless', title='Diff maps'))

    return dats

