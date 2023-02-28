"""
Function(s) reads in monthly data from the single-forcing LENS for selected
variables

Notes
-----
Based on Code by:
    Author : Zachary Labe
    Date   : 8 August 2020
Edited and adjusted by:
    Author : Philine Bommer
    Date   : December 2022

"""
### Import modules
import numpy as np

import warnings
import xarray as xr


from ..utils.utilities_statistics import *
import numpy as np


def open_data(directory, times, vari, varis, simulation, ensmember, i):

    for j in range(int(times.size/2)):
        if simulation == 'CESM1A':
            if vari == 'All':
                if i == 0:
                    if j == 0:
                        time = np.arange(1850, 2005 + 1, 1)
                    elif j > 0:
                        time = np.arange(times[j + 1], times[j + 2] + 1, 1)
                elif i>=35:
                    if j == 0:
                        time = np.arange(times[j], times[j + 1] + 1, 1)
                    elif j > 0:
                        time = np.arange(times[j + 1], 2101 + 1, 1)
                else:
                    if j == 0:
                        time = np.arange(times[j], times[j + 1] + 1, 1)
                    elif j > 0:
                        time = np.arange(times[j + 1], times[j + 2] + 1, 1)
            else:
                if j == 0:
                    time = np.arange(times[j], times[j + 1] + 1, 1)
                else:
                    time = np.arange(times[j + 1], times[j + 2] + 1, 1)
        else:
            if vari == 'All':
                if i == 0:
                    if j == 0:
                        time = np.arange(1850, 2005 + 1, 1)
                    elif j>0:
                        time = np.arange(times[j+1], times[j+2] + 1, 1)

                else:
                    if j == 0:
                        time = np.arange(times[j], times[j + 1] + 1, 1)
                    elif j > 0:
                        time = np.arange(times[j + 1], times[j + 2] + 1, 1)
            else:
                if j == 0:
                    time = np.arange(times[j], times[j+1] + 1, 1)
                else:
                    time = np.arange(times[j+1], times[j + 2] + 1, 1)


        timeslice = '%s-%s' % (time.min(), time.max())
        filename = directory + '/%s_%s%s-%s.nc' % (simulation, vari, ensmember, timeslice)
        data = xr.open_dataset(filename)
        ### Regridding of data
        new_lon = np.arange(data.lon[0], data.lon[-1], 2.5)  # data.dims["lon"] * 4)
        new_lat = np.arange(data.lat[0], data.lat[-1], 1.9)  # data.dims["lat"] * 4)

        data = data.interp(lat=new_lat, lon=new_lon)

        if simulation == 'CESM1':
            if vari == 'All':
                if i==0 :
                    if j ==0:
                        data = data.sel(time = slice('1920-02-01',data.time.values[data.time.values.size-1]))
        elif simulation == 'CESM1A':
            if vari == 'All':
                if i >= 1:
                    if j >0:
                        data = data.sel(time = slice(data.time.values[0],'2081-01-01'))

                elif i==0 :
                    if j ==0:
                        data = data.sel(time = slice('1920-02-01',data.time.values[data.time.values.size-1]))

        # data = Dataset(filename, 'r')

        if j == 0 :
            full_var = data.variables['%s' % varis][:, :, :]
        else:
            full_var = np.concatenate([full_var,data.variables['%s' % varis][:, :, :]], axis = 0)

        lat1 = data.variables['lat'][:]
        lon1 = data.variables['lon'][:]
        data.close()


    return full_var, time, lat1, lon1


def read_LENS(directory, simulation, vari, sliceperiod, slicebase, sliceshape, addclimo, slicenan, takeEnsMean):
    """
    Function reads monthly data from LENS

    Parameters
    ----------
    directory : string
        path for data
    simulation : string
        name of the Forcing LENS
    vari : string
        variable for analysis
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
    takeEnsMean : binary
        whether to take ensemble mean

    Returns
    -------
    lat : 1d numpy array
        latitudes
    lon : 1d numpy array
        longitudes
    var : numpy array
        processed variable
    ENSmean : numpy array
        ensemble mean

    Usage
    -----
    read_SINGLE_LENS(directory,simulation,vari,sliceperiod,
                  slicebase,sliceshape,addclimo,
                  slicenan,takeEnsMean)
    """
    print('\n>>>>>>>>>> STARTING read_SINGLE_LENS function!')


    if simulation == 'CESM1A':
        varis = 'TS'
    else:
        varis = 'TSA'


    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)

    times = np.asarray([1920, 2005, 2006, 2081])

    mon = 12
    if vari == 'All':
        if simulation == 'CESM1':
            nEns = 35
        elif simulation == 'CESM1A':
            nEns = 40
    else:
        nEns = 20

    allens = np.arange(1, nEns + 1, 1)
    ens = list(map('{:d}'.format, allens))

    ### Read in data
    membersvar = []
    for i, ensmember in enumerate(ens):
        full_var, time, lat1, lon1 = open_data(directory, times, vari, varis, simulation, ensmember, i)

        var = full_var
        time = np.arange(times[0], times[times.size - 1], 1)

        print('Completed: read ensemble --%s for %s for %s--' % (simulation, ensmember, vari))
        membersvar.append(var)
        del var

    membersvar = np.asarray(membersvar)

    ensvar = np.reshape(membersvar, (len(ens), time.shape[0], mon,
                                     lat1.shape[0], lon1.shape[0]))

    del membersvar
    print('Completed: read all members!\n')

    ###########################################################################
    ### Calculate anomalies or not
    if addclimo == True:
        ensvalue = ensvar
        print('Completed: calculated absolute variable!')
    elif addclimo == False:
        yearsq = np.where((time >= slicebase.min()) & (time <= slicebase.max()))[0]
        yearssel = time[yearsq]

        mean = np.nanmean(ensvar[:, yearsq, :, :, :])
        ensvalue = ensvar - mean
        print('Completed: calculated anomalies from',
              slicebase.min(), 'to', slicebase.max())

    ###########################################################################
    ### Slice over months (currently = [ens,yr,mn,lat,lon])
    ### Shape of output array
    if sliceperiod == 'annual':
        ensvalue = np.nanmean(ensvalue, axis=2)
        if sliceshape == 1:
            ensshape = ensvalue.ravel()
        elif sliceshape == 4:
            ensshape = ensvalue
        print('Shape of output = ', ensshape.shape, [[ensshape.ndim]])
        print('Completed: ANNUAL MEAN!')

    ###########################################################################
    ### Change missing values
    if slicenan == 'nan':
        ensshape[np.where(np.isnan(ensshape))] = np.nan
        print('Completed: missing values are =', slicenan)
    else:
        ensshape[np.where(np.isnan(ensshape))] = slicenan

    ###########################################################################
    ### Take ensemble mean
    if takeEnsMean == True:
        ENSmean = np.nanmean(ensshape, axis=0)
        print('Ensemble mean AVAILABLE!')
    elif takeEnsMean == False:
        ENSmean = np.nan
        print('Ensemble mean NOT available!')
    else:
        ValueError('WRONG OPTION!')

    ###########################################################################
    ### Change units
    if vari == 'SLP':
        ensshape = ensshape / 100  # Pa to hPa
        ENSmean = ENSmean / 100  # Pa to hPa
        print('Completed: Changed units (Pa to hPa)!')
    elif vari == 'T2M':
        ensshape = ensshape - 273.15  # K to C
        ENSmean = ENSmean - 273.15  # K to C
        print('Completed: Changed units (K to C)!')

    print('>>>>>>>>>> ENDING read_LENS function!')
    print('\n\n\n <<<<<<< --%s ENSEMBLES THROUGH 2080 ANN!!! >>>>>>>\n' % (nEns))
    return lat1, lon1, ensshape, ENSmean