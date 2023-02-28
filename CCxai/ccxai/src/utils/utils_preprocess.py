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
from typing import Dict, List, Any
import cartopy as ct
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pdb

import cphxai.src.utils.utilities_calc as uc

def vis_norm(data : xr.DataArray):
    """
    Case sensitive Min-Max norm for explainability data
    :param datat: raw standardized data
    :return: normed data X ->[-1,1] or X -> [0,1]
    """
    out = data.copy()

    for mod in range(data.model.shape[0]):
        dat = data[{'model': mod}]
        if dat.min() < 0:
            out[{'model': mod}] = ((((dat>0)*dat)/dat.max(dim=['lat','lon'])) -(((dat<0)*dat)/dat.min(dim=['lat','lon'])))
        else:
            out[{'model': mod}] = (((dat > 0) * dat) / dat.max(dim=['lat', 'lon']))


    return out

def xr_raw_norm(datat: xr.DataArray):
    """
    Min-Max norm for raw stndardized data
    :param datat: raw standardized data
    :return: normed data X ->[-1,1]
    """
    out = np.zeros_like(datat.values)
    explanation = datat.values

    if len(explanation.shape) == 6:
        for i in range(datat.shape[0]): # AER, GHG, lens
            for j in range(datat.shape[2]): # Ensemble members
                for k in range(datat.shape[3]): # Images
                    max_val = datat[i,:,j,k,:,:].max().values
                    min_val = datat[i,:,j,k,:,:].min().values
                    xtrema = np.array([max_val,np.abs(min_val)])
                    max_val = np.max(xtrema)
                    counter = 0
                    if min_val < 0.:
                        # positive an negative relevances -> [-1,1]
                        for t in range(datat.shape[1]): # MC samples
                            out[i,t,j,k,:,:] = (explanation[i,t,j,k,:,:] > 0.) * explanation[i,t,j,k,:,:] / max_val - (explanation[i,t,j,k,:,:] <= 0.) * explanation[i,t,j,k,:,:]/ min_val
                            counter += 1
                    else:
                        # only positive relevances -> [0,1]
                        for t in range(datat.shape[1]):
                            out[i,t,j,k,:,:] = (explanation[i,t,j,k,:,:] - min_val) / (max_val - min_val)
                            counter += 1
    elif len(explanation.shape) == 5:
        for i in range(datat.shape[0]): # AER, GHG, lens
            for k in range(datat.shape[2]): # Images
                max_val = datat[i,:,k,:,:].max().values
                min_val = datat[i,:,k,:,:].min().values
                xtrema = np.array([max_val, np.abs(min_val)])
                max_val = np.max(xtrema)
                counter = 0
                if min_val < 0.:
                    # positive and negative relevances -> [-1,1]
                    for t in range(datat.shape[1]): # MC samples
                        out[i,t,k,:,:] = (explanation[i,t,k,:,:] > 0.) * explanation[i,t,k,:,:] / max_val - (explanation[i,t,k,:,:] <= 0.) * explanation[i,t,k,:,:]/ min_val
                        counter += 1
                else:
                    # only positive relevances -> [0,1]
                    for t in range(datat.shape[1]):
                        out[i,t,k,:,:] = (explanation[i,t,k,:,:] - min_val) / (max_val - min_val)
                        counter += 1


    return out


def standardize_relevances(datat: xr.DataArray):
    """
    Standardization for relevances
    :param datat: relevances
    :return: standardized relevances
    """
    out = np.zeros_like(datat.values)
    explanation = datat.values

    if len(explanation.shape) == 6:
        for i in range(datat.shape[0]): # AER, GHG, lens
            for j in range(datat.shape[1]): # Samples
                for k in range(datat.shape[2]): # Ensemble members
                    mean_val = datat[i,j,k,:,:,:].mean().values
                    std_val = datat[i,j,k,:,:,:].std().values
                    counter = 0
                    # only positive relevances -> [0,1]
                    for t in range(datat.shape[3]):
                        out[i,j,k,t,:,:] = (explanation[i,j,k,t,:,:] - mean_val) / (std_val)
                        counter += 1
    elif len(explanation.shape) == 5:
        for i in range(datat.shape[0]): # AER, GHG, lens
            for k in range(datat.shape[1]): # Samples
                mean_val = datat[i,k,:,:,:].mean().values
                std_val = datat[i,k,:,:,:].std().values
                counter = 0
                # only positive relevances -> [0,1]
                for t in range(datat.shape[2]):
                    out[i,k,t,:,:] = (explanation[i,k,t,:,:] - mean_val) / (std_val)
                    counter += 1


    return out


def meancorrect_relevances(datat: xr.DataArray):
    """
    Mean Correction for relevances
    :param datat: relevances
    :return: mean correction relevances
    """
    out = np.zeros_like(datat.values)
    explanation = datat.values

    if len(explanation.shape) == 6:
        for i in range(datat.shape[0]): # AER, GHG, lens
            for j in range(datat.shape[1]): # Samples
                for k in range(datat.shape[2]): # Ensemble members
                    mean_val = datat[i,j,k,:,:,:].mean().values
                    out[i, j, k, :, :] = (explanation[i, j, k, :, :, :] - mean_val)
    elif len(explanation.shape) == 5:
        for i in range(datat.shape[0]): # AER, GHG, lens
            for k in range(datat.shape[1]): # Samples
                mean_val = datat[i,k,:,:,:].mean().values
                out[i,k,:,:,:] = (explanation[i,k,:,:,:] - mean_val)


    return out

def min_max_norm(data: xr.DataArray,
                 dimensions: List):
    """
    Min-max norm relevances:
    :param data: relevance maps
    :param dimensions: dimensions of the array that consider indv. relevance maps (e.g. methods, network samples, years)
    :return: DataArray of relevance snormed [0,1] or [-1,1]
    """
    dtmax = np.abs(data).max(dim=['lat', 'lon'])
    normed = data/dtmax


    return normed

def yrs_inDataset(data, indx, yrs, modelsname):
    """
    Function: constructing dataarray in plot format for plot_xrMaps
    :param data: xr map (4 dimensional)
    :param indx: list of chosen indicies (int arrays)
    :param yrs: list of chosen years (string array)
    :param modelsname: list of first dimension names (string array)
    :return: DataArray
    """


    # vals = data.values
    lats = data['lat'].values
    lons = data['lon'].values
    dats = data[:,indx,:,:]


    catArray = xr.DataArray(data=dats, dims=['models', 'time', 'lat', 'lon'],
                            coords=dict(models=modelsname,
                                        time=yrs, lon=("lon", lons),
                                        lat=("lat", lats)),
                            attrs=dict(description='Diff maps for random sampling of each year',
                                       units='unitless', title='Diff maps'))

    return catArray

def filter_correctPred(data: xr.DataArray,
                       predictions: Any,
                       yrs: Any,
                       idcs: Any,
                       **params):
    """
    Average across explanations of ensemble members per year that where classified in the correct epoch
    :param data: explanations with dim: samp x ensemble x years x lat x lon
    :param predictions: predicted years of the input maps
    :param yrs: true years of inout maps
    :param idcs: ensemble number of each year range
    :param params: dict containing params during training: params['start_year'], params['classChunk'], params['yall'], params['bnd']
    :return: explanations maps with dim: samp x years x lat x lon
    """
    out = data.mean(dim = 'ensemble')

    for smp in range(len(data.samples.values)):
        preds = predictions[smp,:, :, :].reshape(predictions.shape[1]*predictions.shape[2],predictions.shape[3])

        errs = yrs[:, 0] - uc.invert_year_output(preds, params['start_year'], params['classChunk'], params['yall'])
        idxy = np.where(np.abs(errs) <= params['bnd'])
        idxy = idxy[0]
        ensYear = np.zeros([len(idxy), 2])
        ensYear[:, 0] = yrs[idxy, 0]

        k = 0
        l = 0
        for j in range(len(idcs)):
            for i in range(len(params['yall'])):
                if np.abs(errs[k]) <= params['bnd']:
                    ensYear[l, 1] = idcs[j]
                    l += 1
                k += 1

        eYsrt = ensYear[ensYear[:, 0].argsort(), :]
        numEns = np.zeros([len(params['yall']), 2])
        for ys in range(len(params['yall'])):
            indxs = np.where(eYsrt[:, 0] == params['yall'][ys])
            numEns[ys, 0] = params['yall'][ys]
            if np.asarray(indxs).any():
                indexes = eYsrt[indxs, 1][0].astype('int').tolist()
                out[{'samples': smp, 'years': ys}].values = data[{'samples': smp, 'years': ys, 'ensemble': indexes}].mean(dim = 'ensemble').values

    return out

def running_mean(x, N):
    """ x == an array of data. N == number of samples per average """

    cumsum = np.cumsum(np.insert(x, 0, 0))

    return (cumsum[N:] - cumsum[:-N]) / float(N)

def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid
    defined by WGS84

    Input
    ---------
    lat: vector or latitudes in degrees

    Output
    ----------
    r: vector of radius in meters

    Notes
    -----------
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    '''

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b ** 2 / a ** 2)

    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = np.deg2rad(lat)
    lat_gc = np.arctan((1 - e2) * np.tan(lat))

    # radius equation
    # see equation 3-107 in WGS84
    r = (
            (a * (1 - e2) ** 0.5)
            / (1 - (e2 * np.cos(lat_gc) ** 2)) ** 0.5
    )
    return r


def area_grid(lat, lon):
    """
    Calculate the area of each grid cell
    Area is in square meters

    Input
    -----------
    lat: vector of latitude in degrees
    lon: vector of longitude in degrees

    Output
    -----------
    area: grid-cell area in square-meters with dimensions, [lat,lon]

    Notes
    -----------
    Based on the function in
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """


    xlon, ylat = np.meshgrid(lon, lat)
    R = earth_radius(ylat)

    dlat = np.deg2rad(np.gradient(ylat, axis=0))
    dlon = np.deg2rad(np.gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * np.cos(np.deg2rad(ylat))

    area = dy * dx

    xda = xr.DataArray(
        area,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={
            "long_name": "area_per_pixel",
            "description": "area per pixel",
            "units": "m^2",
        },
    )
    return xda

def area_weighting(data: xr.DataArray):
    '''
    Area weighting according to lat-lon grid area variations
    :param data: gridded data
    :return:  weighted data
    '''

    dt_area = area_grid(data.lat, data.lon)
    total_area = dt_area.sum(dim = ['lat', 'lon'])
    data = (data * dt_area) / total_area


    return data

def data_masking(data: xr.DataArray,
                 mask:np.array) -> xr.DataArray:
    """
    Masking of ROI in raw data, with "mean", "uniform", "white" and "black" fill values in the masked data
    in the masked area
    :param data: raw data as DataArray
    :param mask: binary array with masked area indicated by 1
    :return: DataArray with different masking techniques along dimension 'models'
    """

    dats = data.copy()
    row = len(np.unique(np.argwhere(mask)[:,0]))
    col = len(np.unique(np.argwhere(mask)[:,1]))
    arr = data.values
    fill_dict = {
        "mean": float(arr.mean()),
        "uniform":
            np.random.uniform(
                low=0, high=1, size=(row*col,1))
        ,
        "black": float(arr.min()),
        "white": float(arr.max()),
    }
    i = 0

    for masks, vals in fill_dict.items():
        dats = dats.assign_coords({'models':[masks]})
        if isinstance(vals, np.ndarray):
            dats.values[0, 0, mask.astype(dtype=bool)] = vals[:, 0]
        else:
            dats.values[0,0,mask.astype(dtype=bool)] = np.full((row*col,1), vals)[:,0]
        if i == 0:
            maskdata = xr.concat((data,dats), 'models')
        else:
            maskdata = xr.concat((maskdata,dats), 'models')
        i += 1
    return maskdata

def data_pertub(data: xr.DataArray,
                 levels:np.array,) -> xr.DataArray:
    """
    Perturbation of full explanation maps based on additive noise
    :param data: raw data DataArray
    :param levels: array of noise levels
    :return: perturbed data as DataArray and noise levels stacked along 'models' coordinate
    """

    dts = data
    arr = data.values

    i = 0

    for lower_bound in levels:
        pertub = np.random.uniform(low=-lower_bound, high=lower_bound, size=arr.shape)
        dats = data.assign_coords({'models':[lower_bound]})
        dats.values = arr + pertub
        if i == 0:
            pertubdata = xr.concat((dts,dats), 'models')
        else:
            pertubdata = xr.concat((pertubdata,dats), 'models')
        i += 1
    return pertubdata

