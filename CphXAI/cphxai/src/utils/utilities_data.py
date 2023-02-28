"""
Functions are useful utilities for data processing in the NN

Notes
-----
    Author : Zachary Labe
    Date   : 8 July 2020

Usage
-----
    [1] readFiles(variq,dataset)
    [2] getRegion(data,lat1,lon1,lat_bounds,lon_bounds)

"""
import cartopy as ct
import matplotlib.pyplot as plt


# from CphXAI.cphxai2.src.utils.utilities_standard import *
# from CphXAI.cphxai2.src.utils.utilities_calc import *
# from CphXAI.cphxai2.src.read.Read_ENS_data import *
# from CphXAI.cphxai2.src.read.read_20CRv3_monthly import *
# from CphXAI.cphxai2.src.read.read_SMILE import *
# from CphXAI.cphxai2.src.read.read_randomData_monthly import *

from ..read.Read_ENS_data import *
from ..read.read_20CRv3_monthly import *
from ..read.read_SMILE import *
from ..read.read_randomData_monthly import *


def readFiles(variq, dataset, monthlychoice, dirs):
    """
    Function reads in data for selected dataset

    Parameters
    ----------
    variq : string
        variable for analysis
    dataset : string
        name of data set for primary data

    Returns
    -------
    data : numpy array
        data from selected data set
    lat1 : 1d numpy array
        latitudes
    lon1 : 1d numpy array
        longitudes

    Usage
    -----
    data,lat1,lon1 = readFiles(variq,dataset)
    """
    print('\n>>>>>>>>>> Using readFiles function!')

    if dataset == 'lens':

        directorydataLL = dirs + 'LENS/monthly/'
        slicebaseLL = np.arange(1951, 1980 + 1, 1)
        sliceshapeLL = 4
        simulationLL = 'CESM1A'
        variq = 'All'
        slicenanLL = 'nan'
        addclimoLL = True
        takeEnsMeanLL = False
        lat1, lon1, data, ENSmean = read_LENS(directorydataLL, simulationLL, variq,
                                                 monthlychoice, slicebaseLL,
                                                 sliceshapeLL, addclimoLL,
                                                 slicenanLL, takeEnsMeanLL)

    elif dataset == '20CRv3':

        directorydataTW = dirs + '20CRv3/'
        sliceyearTW = np.arange(1836, 2015 + 1, 1)
        sliceshapeTW = 3
        slicenanTW = 'nan'
        addclimoTW = True
        ENSmean = np.nan
        lat1, lon1, data = read_20CRv3_monthly(variq, directorydataTW,
                                                  monthlychoice, sliceyearTW,
                                                  sliceshapeTW, addclimoTW,
                                                  slicenanTW)
    # elif dataset == 'RANDOM':
    #
    #     directorydataRA = dirs
    #     slicebaseRA = np.arange(1951, 1980 + 1, 1)
    #     sliceshapeRA = 4
    #     slicenanRA = 'nan'
    #     addclimoRA = True
    #     takeEnsMeanRA = False
    #     lat1, lon1, data, ENSmean = read_randomData_monthly(directorydataRA, variq,
    #                                                            monthlychoice, slicebaseRA,
    #                                                            sliceshapeRA, addclimoRA,
    #                                                            slicenanRA, takeEnsMeanRA)
    elif any([dataset == 'GHG', dataset == 'AER']):

        directorySI = dirs + 'LENS/monthly/'
        simulationSI = 'CESM1A'
        variq = dataset
        slicebaseSI = np.arange(1951, 1980 + 1, 1)
        sliceshapeSI = 4
        slicenanSI = 'nan'
        addclimoSI = True
        takeEnsMeanSI = False
        # if dataset == 'AER':
        #     pdb.set_trace()
        lat1, lon1, data, ENSmean = read_LENS(directorySI, simulationSI, variq, monthlychoice,
                                                        slicebaseSI, sliceshapeSI, addclimoSI,
                                                        slicenanSI, takeEnsMeanSI)
    else:
        ValueError('WRONG DATA SET SELECTED!')

    print('>>>>>>>>>> Completed: Finished readFiles function!')
    return data, lat1, lon1


def read_primary_dataset(variq, dataset, lat_bounds, lon_bounds, monthlychoice, dirs):
    data, lats, lons = readFiles(variq, dataset, monthlychoice, dirs)
    datar, lats, lons = getRegion(data, lats, lons, lat_bounds, lon_bounds)
    print('\nOur dataset: ', dataset, ' is shaped', data.shape)
    return datar, lats, lons


def read_obs_dataset(variq, dataset_obs, lat_bounds, lon_bounds, monthlychoice, yearsall, sis, dirs):
    data_obs, lats_obs, lons_obs = readFiles(variq, dataset_obs, monthlychoice, dirs)
    data_obs, lats_obs, lons_obs = getRegion(data_obs, lats_obs, lons_obs,
                                                lat_bounds, lon_bounds)
    if dataset_obs == '20CRv3':
        if monthlychoice == 'DJF':
            year20cr = np.arange(1837, 2015 + 1)
        else:
            year20cr = np.arange(1836, 2015 + 1)
        year_obsall = np.arange(yearsall[sis].min(), yearsall[sis].max() + 1, 1)
        yearqq = np.where((year20cr >= year_obsall.min()) & (year20cr <= year_obsall.max()))[0]
        data_obs = data_obs[yearqq, :, :]

    print('our OBS dataset: ', dataset_obs, ' is shaped', data_obs.shape)
    return data_obs, lats_obs, lons_obs


def getRegion(data, lat1, lon1, lat_bounds, lon_bounds):
    """
    Function masks out region for data set

    Parameters
    ----------
    data : 3d+ numpy array
        original data set
    lat1 : 1d array
        latitudes
    lon1 : 1d array
        longitudes
    lat_bounds : 2 floats
        (latmin,latmax)
    lon_bounds : 2 floats
        (lonmin,lonmax)

    Returns
    -------
    data : numpy array
        MASKED data from selected data set
    lat1 : 1d numpy array
        latitudes
    lon1 : 1d numpy array
        longitudes

    Usage
    -----
    data,lats,lons = getRegion(data,lat1,lon1,lat_bounds,lon_bounds)
    """
    print('\n>>>>>>>>>> Using get_region function!')



    ### Note there is an issue with 90N latitude (fixed!)
    lat1 = np.round(lat1, 3)
    ### Mask latitudes
    if data.ndim == 3:
        latq = np.where((lat1 >= lat_bounds[0]) & (lat1 <= lat_bounds[1]))[0]
        latn = lat1[latq]
        datalatq = data[:, latq, :]
        ### Mask longitudes
        lonq = np.where((lon1 >= lon_bounds[0]) & (lon1 <= lon_bounds[1]))[0]
        lonn = lon1[lonq]
        datalonq = datalatq[:, :, lonq]

    elif data.ndim == 4:
        latq = np.where((lat1 >= lat_bounds[0]) & (lat1 <= lat_bounds[1]))[0]
        latn = lat1[latq]
        datalatq = data[:, :, latq, :]
        ### Mask longitudes
        lonq = np.where((lon1 >= lon_bounds[0]) & (lon1 <= lon_bounds[1]))[0]
        lonn = lon1[lonq]
        datalonq = datalatq[:, :, :, lonq]

    elif data.ndim == 6:
        latq = np.where((lat1 >= lat_bounds[0]) & (lat1 <= lat_bounds[1]))[0]
        latn = lat1[latq]
        datalatq = data[:, :, :, latq, :]
        ### Mask longitudes
        lonq = np.where((lon1 >= lon_bounds[0]) & (lon1 <= lon_bounds[1]))[0]
        lonn = lon1[lonq]
        datalonq = datalatq[:, :, :, :, lonq]

    ### New variable name
    datanew = datalonq

    print('>>>>>>>>>> Completed: getRegion function!')
    return datanew, latn, lonn



def dataarrayLENS(lats, lons, var, directory, SAMPLEQ):
    print('\n>>> Using netcdf4LENS function!')



    name = 'LRP_YearlyMaps_%s_20ens_T2M_annual.nc' % SAMPLEQ
    filename = directory + name
    # ncfile = Dataset(filename,'w',format='NETCDF4')
    # ncfile.description = 'LRP maps for observations for each model (annual, selected seed)'

    ### Data

    lrpnc = xr.DataArray(data=var, dims=['model', 'samples', 'years','lat','lon'],
                         coords=dict(model= np.arange(var.shape[0]), samples=np.arange(var.shape[1]), years=np.arange(var.shape[2]), lon=("lon", lons),
                                     lat=("lat", lats)),
                         attrs=dict(
                             description='LRP maps for random sampling of each year',
                             units='unitless relevance', title = 'LRP relevance'))

    lrpnc.to_netcdf(filename)

    print('*Completed: Created netCDF4 File!')

def drawOnGlobe(axes, data, lats, lons, region, cmap='coolwarm', vmin=None, vmax=None, inc=None):
    '''Usage: drawOnGlobe(data, lats, lons, basemap, cmap)
          data: nLats x nLons
          lats: 1 x nLats
          lons: 1 x nLons
          basemap: returned from getRegion
          cmap
          vmin
          vmax'''

    data_cyc, lons_cyc = data, lons
    ### Fixes white line by adding point

    data_cyc, lons_cyc = ct.util.add_cyclic_point(data, coord=lons)

    image = plt.pcolormesh(lons_cyc, lats, data_cyc, transform=ct.crs.PlateCarree(),
                           vmin=vmin, vmax=vmax, cmap=cmap, shading='flat')

    axes.coastlines(color='black', linewidth=1.2)

    divider = make_axes_locatable(axes)
    ax_cb = divider.new_horizontal(size="2%", pad=0.1, axes_class=plt.Axes)

    plt.gcf().add_axes(ax_cb)
    #     if vmin is not None:
    #       cb = plt.colorbar(image, cax=ax_cb,
    #                           boundaries=np.arange(vmin,vmax+inc,inc))
    #     else:
    #       cb = plt.colorbar(image, cax=ax_cb)
    cb = plt.colorbar(image, cax=ax_cb)
    cb.set_label('units', fontsize=20)

    plt.sca(axes)  # in case other calls, like plt.title(...), will be made

    ### Return image
    return cb, image



def segment_data(data, fac, sis, yearsall, debug, random_segment_seed):


    if random_segment_seed == None:
        random_segment_seed = int(int(np.random.randint(1, 100000)))
    np.random.seed(random_segment_seed)

    if fac < 1:
        nrows = data.shape[0]
        segment_train = int(np.round(nrows * fac))
        segment_test = nrows - segment_train
        print('Training on', segment_train, 'ensembles, testing on', segment_test)

        ### Picking out random ensembles
        i = 0
        trainIndices = list()
        while i < segment_train:
            line = np.random.randint(0, nrows)
            if line not in trainIndices:
                trainIndices.append(line)
                i += 1
            else:
                pass

        i = 0
        testIndices = list()
        while i < segment_test:
            line = np.random.randint(0, nrows)
            if line not in trainIndices:
                if line not in testIndices:
                    testIndices.append(line)
                    i += 1
            else:
                pass

        ### Random ensembles are picked
        if debug:
            print('Training on ensembles: ', trainIndices)
            print('Testing on ensembles: ', testIndices)

        ### Training segment----------
        data_train = ''
        for ensemble in trainIndices:
            this_row = data[ensemble, :, :, :]
            this_row = this_row.reshape(-1, data.shape[1], data.shape[2],
                                        data.shape[3])
            if data_train == '':
                data_train = np.empty_like(this_row)
            data_train = np.vstack((data_train, this_row))
        data_train = data_train[1:, :, :, :]

        if debug:
            print('org data - shape', data.shape)
            print('training data - shape', data_train.shape)

        ### Reshape into X and T
        Xtrain = data_train.reshape((data_train.shape[0] * data_train.shape[1]),
                                    (data_train.shape[2] * data_train.shape[3]))
        Ttrain = np.tile(
            (np.arange(data_train.shape[1]) + yearsall[sis].min()).reshape(data_train.shape[1], 1),
            (data_train.shape[0], 1))
        Xtrain_shape = (data_train.shape[0], data_train.shape[1])

        ### Testing segment----------
        data_test = ''
        for ensemble in testIndices:
            this_row = data[ensemble, :, :, :]
            this_row = this_row.reshape(-1, data.shape[1], data.shape[2],
                                        data.shape[3])
            if data_test == '':
                data_test = np.empty_like(this_row)
            data_test = np.vstack((data_test, this_row))
        data_test = data_test[1:, :, :, :]

        if debug:
            print('testing data', data_test.shape)

        ### Reshape into X and T
        Xtest = data_test.reshape((data_test.shape[0] * data_test.shape[1]),
                                  (data_test.shape[2] * data_test.shape[3]))
        Ttest = np.tile(
            (np.arange(data_test.shape[1]) + yearsall[sis].min()).reshape(data_test.shape[1], 1),
            (data_test.shape[0], 1))

    else:
        trainIndices = np.arange(0, np.shape(data)[0])
        testIndices = np.arange(0, np.shape(data)[0])
        print('Training on ensembles: ', trainIndices)
        print('Testing on ensembles: ', testIndices)

        data_train = data
        data_test = data

        Xtrain = data_train.reshape((data_train.shape[0] * data_train.shape[1]),
                                    (data_train.shape[2] * data_train.shape[3]))
        Ttrain = np.tile(
            (np.arange(data_train.shape[1]) + yearsall[sis].min()).reshape(data_train.shape[1], 1),
            (data_train.shape[0], 1))
        Xtrain_shape = (data_train.shape[0], data_train.shape[1])

    Xtest = data_test.reshape((data_test.shape[0] * data_test.shape[1]),
                              (data_test.shape[2] * data_test.shape[3]))
    Ttest = np.tile((np.arange(data_test.shape[1]) + yearsall.min()).reshape(data_test.shape[1], 1),
                    (data_test.shape[0], 1))

    Xtest_shape = (data_test.shape[0], data_test.shape[1])
    data_train_shape = data_train.shape[1]
    data_test_shape = data_test.shape[1]

    ### 'unlock' the random seed
    np.random.seed(None)

    return Xtrain, Ttrain, Xtest, Ttest, Xtest_shape, Xtrain_shape, data_train_shape, data_test_shape, testIndices, trainIndices, random_segment_seed


def shape_obs(data_obs, year_obs):
    Xtest_obs = np.reshape(data_obs, (data_obs.shape[0],
                                      (data_obs.shape[1] * data_obs.shape[2])))
    Ttest_obs = np.tile(np.arange(data_obs.shape[0]) + year_obs[0])
    return Xtest_obs, Ttest_obs

def shape_data(X,data):
    ''' Reshapes flattened data into images for image classification (CNN)'''
    # pdb.set_trace()
    if len(data.shape) >3:
        X = X.reshape((X.shape[0],data.shape[2],data.shape[3],1))
    else:
        X = X.reshape(((X.shape[0]), data.shape[1],data.shape[2],1))
    return X

def one_hot_encoded(Y):
    ''' Provides one-hot-encoded labels for CNN on basis of fuzzy labels (Labe et. al.)'''
    # pdb.set_trace()
    labels = np.zeros((Y.shape))
    for i in range(Y.shape[0]):
        cl = np.argmax(Y[i,:])
        labels[i,cl] = 1

    return labels

def findStringMiddle(start, end, s):
    return s[s.find(start) + len(start):s.rfind(end)]

def dataarrayLENS(lats, lons, var, directory, SAMPLEQ, xtype):
    '''
    Function for LRP maps saving in xarray data array format
     var: ndarray of shape #climate models x # sampled models x time x lat x lons
     SAMPLEQ: #sampled models
     directory:
     '''

    import xarray as xr
    import numpy as np

    name = '%s_YearlyMaps_%s_20ens_T2M_annual.nc' % (xtype,SAMPLEQ)
    filename = directory + name


    lrpnc = xr.DataArray(data=var, dims=['model', 'samples', 'years','lat','lon'],
                         coords=dict(model= np.arange(var.shape[0]), samples=np.arange(var.shape[1]), years=np.arange(var.shape[2]), lon=("lon", lons),
                                     lat=("lat", lats)),
                         attrs=dict(
                             description='LRP maps for random sampling of each year',
                             units='unitless relevance', title = 'LRP relevance'))
    lrpnc.to_netcdf(filename)

    print('*Completed: Created netCDF4 File!')

def xrconcat(iterators, path, ax):
    '''
    Function: concatenate dataarrays from indv. path
    along axis of choice
    ax: num. of axis (int)
    path: data path (string)
    iterators: filenames (list of string)
     '''

    for i in range(len(iterators)):
        data = xr.open_data_array(path + iterators[i])
        if i == 0:
            catArray = data
        else:
            catArray = xr.concat((catArray,data),ax)

    return catArray

def dataarrayUAI(lats, lons, var, directory, SAMPLEQ, type, dty):
    '''
    Function for UAI maps saving in xarray data array format
     var: ndarray of shape #climate models x # sampled models x time x lat x lons
     SAMPLEQ: #sampled models
     directory:
     '''
    print('\n>>> Using netcdf4LENS function!')

    import xarray as xr
    import numpy as np

    name = 'UAI_YearlyMaps_%s_20ens_T2M_%s_%s_annual.nc' % (SAMPLEQ,dty,type)
    filename = directory + name

    ### Data
    if len(var.shape) == 6:
        lrpnc = xr.DataArray(data=var, dims=['model', 'samples', 'ensemble', 'years','lat','lon'],
                             coords=dict(model= np.arange(var.shape[0]), samples=np.arange(var.shape[1]), ensemble=np.arange(var.shape[2]), years=np.arange(var.shape[3]), lon=("lon", lons),
                                         lat=("lat", lats)),
                             attrs=dict(
                                 description='LRP maps for random sampling of each year',
                                 units='unitless relevance', title = 'LRP relevance'))
    else:
        lrpnc = xr.DataArray(data=var, dims=['model', 'samples', 'years', 'lat', 'lon'],
                             coords=dict(model=np.arange(var.shape[0]), samples=np.arange(var.shape[1]),
                                         years=np.arange(var.shape[2]),
                                         lon=("lon", lons),
                                         lat=("lat", lats)),
                             attrs=dict(
                                 description='LRP maps for random sampling of each year',
                                 units='unitless relevance', title='LRP relevance'))

    lrpnc.to_netcdf(filename)

    print('*Completed: Created netCDF4 File!')


def yrs_inDataset(data, indx, yrs, modelsname):
    '''
    Function:
    constructing dataarray in plot format for plot_xrMaps
    data: xr map (4 dimensional)
    indx: list of chosen indicies (int arrays)
    yrs: list of chosen years (string array)
    modelsname: list of first dimension names (string array)
     '''

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






