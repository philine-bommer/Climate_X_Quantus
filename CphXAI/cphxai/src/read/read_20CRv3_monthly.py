"""
Function reads in monthly data from 20CRv3

Notes
-----
    Author : Zachary Labe
    Date   : 15 July 2020

Usage
-----
    [1] read_20CRv3_monthly(variq,directory,sliceperiod,sliceyear,
                  sliceshape,addclimo,slicenan)
"""

# from CphXAI.cphxai2.src.utils.utilities_standard import *
from ..utils.utilities_statistics import *
import xarray as xr
import numpy as np

def read_20CRv3_monthly(variq, directory, sliceperiod, sliceyear, sliceshape, addclimo, slicenan):
    """
    Function reads monthly data from 20CRv3

    Parameters
    ----------
    variq : string
        variable to retrieve
    directory : string
        path for data
    sliceperiod : string
        how to average time component of data
    sliceyear : string
        how to slice number of years for data
    sliceshape : string
        shape of output array
    addclimo : binary
        True or false to add climatology
    slicenan : string or float
        Set missing values

    Returns
    -------
    lat : 1d numpy array
        latitudes
    lon : 1d numpy array
        longitudes
    var : 3d numpy array or 4d numpy array
        [time,lat,lon] or [year,month,lat,lon]

    Usage
    -----
    lat,lon,var = read_20CRv3_monthly(variq,directory,sliceperiod,sliceyear,
                            sliceshape,addclimo,slicenan)
    """
    print('\n>>>>>>>>>> STARTING read_20CRv3_monthly function!')

    ### Parameters
    time = np.arange(1836, 2015 + 1, 1)
    monthslice = sliceyear.shape[0] * 12
    mon = 12

    ###########################################################################
    ### Read in data
    filename = 'monthly/%s_1836-2015.nc' % variq
    data = xr.open_dataset(directory + filename)
    ### Regridding of data
    new_lon = np.arange(data.lon[0], data.lon[-1], 2.5)#data.dims["lon"] * 4)
    new_lat = np.arange(data.lat[0], data.lat[-1], 1.9)#data.dims["lat"] * 4)

    data = data.interp(lat=new_lat, lon=new_lon)
    ### Read file
    lat1 = data.variables['lat'][:]
    lon1 = data.variables['lon'][:]
    variq = 'air'
    var = data.variables['%s' % variq][-monthslice:, :, :]
    data.close()
    variq = 'T2M'
    print('Years of output =', sliceyear.min(), 'to', sliceyear.max())
    ###########################################################################
    ### Reshape data into [year,month,lat,lon]
    datamon = np.reshape(var.values, (var.shape[0] // mon, mon,
                                      lat1.shape[0], lon1.shape[0]))

    ###########################################################################
    ### Return absolute temperature (1951-1980 baseline)
    if addclimo == True:
        varmon = datamon
        print('Completed: calculated absolute variable!')
    else:
        yearbasemin = 1981
        yearbasemax = 2010
        yearq = np.where((time >= yearbasemin) & (time <= yearbasemax))[0]
        varmon = datamon - np.nanmean(datamon[yearq, :, :, :], axis=0) # monthly temp - yearly average
        print('Completed: calculated anomalies!')

    ###########################################################################
    ### Slice over months (currently = [yr,mn,lat,lon])
    ### Shape of output array
    if sliceperiod == 'annual':
        vartime = np.nanmean(varmon, axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif sliceshape == 3:
            varshape = vartime
        print('Shape of output = ', varshape.shape, [[varshape.ndim]])
        print('Completed: ANNUAL MEAN!')
    elif sliceperiod == 'DJF':
        varshape = calcDecJanFeb(varmon, lat1, lon1, 'surface', 1)
        print('Shape of output = ', varshape.shape, [[varshape.ndim]])
        print('Completed: DJF MEAN!')
    elif sliceperiod == 'MAM':
        vartime = np.nanmean(varmon[:, 2:5, :, :], axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif sliceshape == 3:
            varshape = vartime
        print('Shape of output = ', varshape.shape, [[varshape.ndim]])
        print('Completed: MAM MEAN!')
    elif sliceperiod == 'JJA':
        vartime = np.nanmean(varmon[:, 5:8, :, :], axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif sliceshape == 3:
            varshape = vartime
        print('Shape of output = ', varshape.shape, [[varshape.ndim]])
        print('Completed: JJA MEAN!')
    elif sliceperiod == 'SON':
        vartime = np.nanmean(varmon[:, 8:11, :, :], axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif sliceshape == 3:
            varshape = vartime
        print('Shape of output = ', varshape.shape, [[varshape.ndim]])
        print('Completed: SON MEAN!')
    elif sliceperiod == 'JFM':
        vartime = np.nanmean(varmon[:, 0:3, :, :], axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif sliceshape == 3:
            varshape = vartime
        print('Shape of output = ', varshape.shape, [[varshape.ndim]])
        print('Completed: JFM MEAN!')
    elif sliceperiod == 'AMJ':
        vartime = np.nanmean(varmon[:, 3:6, :, :], axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif sliceshape == 3:
            varshape = vartime
        print('Shape of output = ', varshape.shape, [[varshape.ndim]])
        print('Completed: AMJ MEAN!')
    elif sliceperiod == 'JAS':
        vartime = np.nanmean(varmon[:, 6:9, :, :], axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif sliceshape == 3:
            varshape = vartime
        print('Shape of output = ', varshape.shape, [[varshape.ndim]])
        print('Completed: JAS MEAN!')
    elif sliceperiod == 'OND':
        vartime = np.nanmean(varmon[:, 9:, :, :], axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif sliceshape == 3:
            varshape = vartime
        print('Shape of output = ', varshape.shape, [[varshape.ndim]])
        print('Completed: OND MEAN!')
    elif sliceperiod == 'none':
        vartime = varmon
        if sliceshape == 1:
            varshape = varshape.ravel()
        elif sliceshape == 3:
            varshape = np.reshape(vartime, (vartime.shape[0] * vartime.shape[1],
                                            vartime.shape[2], vartime.shape[3]))
        elif sliceshape == 4:
            varshape = varmon
        print('Shape of output =', varshape.shape, [[varshape.ndim]])
        print('Completed: ALL MONTHS!')

    ###########################################################################
    ### Change missing values
    if slicenan == 'nan':
        varshape[np.where(np.isnan(varshape))] = np.nan
        print('Completed: missing values are =', slicenan)
    else:
        varshape[np.where(np.isnan(varshape))] = slicenan

    ###########################################################################
    ### Change units
    if variq == 'SLP':
        varshape = varshape / 100  # Pa to hPa
        print('Completed: Changed units (Pa to hPa)!')
    elif variq == 'T2M':
        varshape = varshape - 273.15  # K to C
        print('Completed: Changed units (K to C)!')

    print('>>>>>>>>>> ENDING read_20CRv3_monthly function!')

    return lat1, lon1, varshape