### Import packages
from typing import Tuple, Optional, Any, List
import palettable.cubehelix as cm
import matplotlib as mpl
import numpy as np
import xarray as xr
import yaml
import os
import sys
import pdb
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


import cphxai.src.utils.utilities_data as ud


def list_files_from_seeds(pat, seeds):
    fil = os.listdir(pat)
    fils=np.asarray(fil)
    files = []
    for i in range(len(fils)):
        fullstring = fils[i]
        if str(seeds[i]) in fullstring:
            files.append(fullstring)
    return files


def list_multisubs(filelist, datasingle,  string1, string2):

    sublisth5 = []
    sublistnpz = []
    for i in range(len(filelist)):
        fullstring = filelist[i]
        if datasingle in fullstring and string1 in fullstring:
            sublisth5.append(fullstring)
        if datasingle in fullstring and string2 in fullstring and 'modelTrainTest' in fullstring:
            sublistnpz.append(fullstring)

    return sublisth5, sublistnpz

def list_files(pat):
    """
    list files in folder
    :param pat: folder path
    :return: list of all files within folder
    """
    fil = os.listdir(pat)
    files=np.asarray(fil)
    return files


def list_subs(pat: str(),
              sub: str(),
              **kwargs):
    """

    :param pat: path to files
    :param sub: specific string in filename
    :return: list of files including string
    """
    if kwargs is None:
        kwargs = {}
    sublist = []
    filelist = list_files(pat)
    settings = kwargs.get('settings',[])

    for i in range(len(filelist)):
        fullstring = filelist[i]
        if sub in fullstring:
            if len(settings) >= 1:
                for set in settings:
                    if set in fullstring:
                        sublist.append(fullstring)
            else:
                sublist.append(fullstring)

    return sublist

def sortfilelist(filelist: str,
                 ** params):
    """
    Sorts list of files according to wished order (params - order)
    :param filelist: list of filenames
    :param params: order - List of strings in preferred order
    :return: sorted list with filenames
    """
    order = params['order']
    filesort = []

    for i in range(len(order)):
        dp = 0
        for j in range(len(filelist)):
            orders = order[i] + '_'
            if orders in filelist[j] and filelist[j] not in filesort:
                if dp <1:
                    filesort.append(filelist[j])
                dp += 1

    return filesort

def data_concatenate(filelist: Any,
                     directory: Any,
                     dimension: str,
                     ** params):
    """
    Concatenate loaded dataset
    :param filelist: list of files to be loaded
    :param directory: list of according directories
    :param dimension: string of data dimension to concatenate along
    :param params dict
    :return: DataArray
    """
    for i in range(len(filelist)):
        if i == 0:
            data = xr.open_dataarray(directory[i] + filelist[i])
            try:
                if params['ens'] == 'mean':
                    data = data.mean(dim= 'ensemble', skipna=True)  # .mean(dim = 'ensemble')
                else:
                    data = data[{'ensemble':params['ens']}]#.mean(dim = 'ensemble')

            except:
                print('Ensembles are averaged')
        else:
            dds = xr.open_dataarray(directory[i] + filelist[i])
            try:
                if params['ens'] == 'mean':
                    dds = dds.mean(dim= 'ensemble', skipna=True)  # .mean(dim = 'ensemble')
                else:
                    dds = dds[{'ensemble':params['ens']}]#.mean(dim = 'ensemble')
            except:
                print('Ensembles are averaged')
            data = xr.concat((data, dds), dimension)
    return data

def data_concat_ens(filelist: Any,
                     directory: Any,
                     dimension: str,
                     ** params):
    """
    Concatenate loaded datasets but maintains ensembles instead of samples
    :param filelist: list of files to be loaded
    :param directory: list of according directories
    :param dimension: string of data dimension to concatenate along
    :param params dict
    :return: DataArray
    """
    for i in range(len(filelist)):
        if i == 0:
            data = xr.open_dataarray(directory[i] + filelist[i])
            try:
                data = data[{'samples':params['ens']}]#.mean(dim = 'ensemble')

            except:
                print('Wrong Data')
        else:
            dds = xr.open_dataarray(directory[i] + filelist[i])
            try:
                dds = dds[{'samples':params['ens']}]#.mean(dim = 'ensemble')
            except:
                print('Wrong Data')
            data = xr.concat((data, dds), dimension)
    return data


def raw_data(variq, **params):
    """
    Load raw data
    :param variq: string - T2M  (data type)
    :param params: set of params of the data loaded
    :return: DataArray of raw daat
    """
    years = np.arange(params['start_year'], params['end_year'] + 1, 1)
    monthlychoice = params['seasons'][0]
    reg_name = params['reg_name']
    dirdata = params['dirdata']


    if params['end_year'] > 2020:
        lat_bounds, lon_bounds = ud.regions(reg_name)
        data_all, lats, lons = ud.read_primary_dataset(variq, params['datafiles'],
                                                       lat_bounds,
                                                       lon_bounds, monthlychoice, dirdata)

        dataS = data_all[np.newaxis,:,:,:,:]

        catArray = xr.DataArray(data=dataS, dims=['model', 'ensembles', 'years', 'lat', 'lon'],
                                coords=dict(model=np.asarray(['Raw']), ensembles=np.arange(dataS.shape[1]),
                                            years=years, lon=("lon", lons),
                                            lat=("lat", lats)),
                                attrs=dict(description='Diff maps for random sampling of each year',
                                           units='unitless', title='Diff maps'))

        Xmean = catArray.mean(dim = ['model','ensembles','years'], skipna = True)
        Xstd = catArray.std(dim = ['model','ensembles','years'], skipna = True)

        catArray = (catArray - Xmean) / Xstd


    else:
        lat_bounds, lon_bounds = ud.regions(reg_name)
        data_all, lats, lons = ud.read_obs_dataset(variq, params['dataset_obs'],
                                                               lat_bounds,
                                                               lon_bounds, monthlychoice, [years], 0, dirdata)
        Xmeanobs = np.nanmean(data_all, axis=0)
        Xstdobs = np.nanstd(data_all, axis=0)

        data_allS = (data_all - Xmeanobs) / Xstdobs
        data_allS[np.isnan(data_allS)] = 0
        dataS = data_allS[np.newaxis,np.newaxis,:,:,:]


        catArray = xr.DataArray(data=dataS, dims=['model', 'ensembles','years', 'lat', 'lon'],
                                coords=dict(model=np.asarray(['Raw']), ensembles= np.arange(dataS.shape[1]),
                                            years=years, lon=("lon", lons),
                                            lat=("lat", lats)),
                                attrs=dict(description='Diff maps for random sampling of each year',
                                           units='unitless', title='Diff maps'))

    return catArray