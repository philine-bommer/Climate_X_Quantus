
import numpy as np
import scipy.stats as stats
import copy as copy
import os
import sys
import pdb
from ..utils.utilities_statistics import *
from ..utils.utilities_calc import *



"""
Functions are useful statistical untilities for data processing in the NN

Notes
-----
    Author : Zachary Labe
    Date   : 15 July 2020

Usage
-----
    [1] rmse(a,b)
    [2] remove_annual_mean(data,data_obs,lats,lons,lats_obs,lons_obs)
    [3] remove_merid_mean(data,data_obs)
    [4] remove_ensemble_mean(data)
    [5] remove_ocean(data,data_obs)
    [6] remove_land(data,data_obs)
    [7] standardize_data(Xtrain,Xtest)
"""


def rmse(a, b):
    """calculates the root mean squared error
    takes two variables, a and b, and returns value
    """

    ### Import modules
    import numpy as np

    ### Calculate RMSE
    rmse_stat = np.sqrt(np.mean((a - b) ** 2))

    return rmse_stat


def remove_annual_mean(data, data_obs, lats, lons, lats_obs, lons_obs):
    """
    Removes annual mean from data set
    """

    ### Import modulates


    ### Create 2d grid
    lons2, lats2 = np.meshgrid(lons, lats)
    lons2_obs, lats2_obs = np.meshgrid(lons_obs, lats_obs)

    ### Calculate weighted average and remove mean
    data = data - calc_weightedAve(data, lats2)[:, :, np.newaxis, np.newaxis]
    data_obs = data_obs - calc_weightedAve(data_obs, lats2_obs)[:, np.newaxis, np.newaxis]

    return data, data_obs


def remove_merid_mean(data, data_obs):
    """
    Removes annual mean from data set
    """

    ### Import modulates
    import numpy as np

    ### Move mean of latitude
    data = data - np.nanmean(data, axis=2)[:, :, np.newaxis, :]
    data_obs = data_obs - np.nanmean(data_obs, axis=1)[:, np.newaxis, :]

    return data, data_obs


def remove_ensemble_mean(data):
    """
    Removes ensemble mean
    """

    ### Import modulates
    import numpy as np

    ### Remove ensemble mean
    datameangone = data - np.nanmean(data, axis=0)

    return datameangone


def remove_ocean(data, data_obs):
    """
    Masks out the ocean for land_only == True
    """

    ### Import modules
    # from netCDF4 import Dataset
    import xarray as xr

    ### Read in land mask
    directorydata = '/Users/zlabe/Data/masks/'
    filename = 'lsmask_19x25.nc'
    datafile = xr.open_dataset(directorydata + filename)
    mask = datafile.variables['nmask'][:]
    datafile.close()

    ### Mask out model and observations
    datamask = data * mask
    data_obsmask = data_obs * mask

    return datamask, data_obsmask


def remove_land(data, data_obs):
    """
    Masks out the ocean for ocean_only == True
    """

    ### Import modules
    from netCDF4 import Dataset

    ### Read in ocean mask
    directorydata = '/Users/zlabe/Data/masks/'
    filename = 'ocmask_19x25.nc'
    datafile = Dataset(directorydata + filename)
    mask = datafile.variables['nmask'][:]
    datafile.close()

    ### Mask out model and observations
    datamask = data * mask
    data_obsmask = data_obs * mask

    return datamask, data_obsmask


def standardize_data(Xtrain, Xtest):
    """
    Standardizes training and testing data
    """

    ### Import modulates
    import numpy as np

    Xmean = np.nanmean(Xtrain, axis=0)
    Xstd = np.nanstd(Xtrain, axis=0)
    Xtest = (Xtest - Xmean) / Xstd
    Xtrain = (Xtrain - Xmean) / Xstd

    stdVals = (Xmean, Xstd)
    stdVals = stdVals[:]

    return Xtrain, Xtest, stdVals


def convert_fuzzyDecade(data, startYear, classChunk, yearsall):

    years = np.arange(startYear - classChunk * 2, yearsall.max() + classChunk * 2)
    chunks = years[::int(classChunk)] + classChunk / 2

    labels = np.zeros((np.shape(data)[0], len(chunks)))

    for iy, y in enumerate(data):
        norm = stats.uniform.pdf(years, loc=y - classChunk / 2., scale=classChunk)

        vec = []
        for sy in years[::classChunk]:
            j = np.logical_and(years > sy, years < sy + classChunk)
            vec.append(np.sum(norm[j]))
        vec = np.asarray(vec)
        vec[vec < .0001] = 0.  # This should not matter

        vec = vec / np.sum(vec)

        labels[iy, :] = vec
    return labels, chunks


def convert_fuzzyDecade_toYear(label, startYear, classChunk, yearsall):


    print('SELECT END YEAR - HARD CODED IN FUNCTION')
    years = np.arange(startYear - classChunk * 2, yearsall.max() + classChunk * 2)
    chunks = years[::int(classChunk)] + classChunk / 2

    return np.sum(label * chunks, axis=1)


def invert_year_outputChunk(ypred, startYear, classChunkHalf):


    if (len(np.shape(ypred)) == 1):
        maxIndices = np.where(ypred == np.max(ypred))[0]
        if (len(maxIndices) > classChunkHalf):
            maxIndex = maxIndices[classChunkHalf]
        else:
            maxIndex = maxIndices[0]

        inverted = maxIndex + startYear - classChunkHalf

    else:
        inverted = np.zeros((np.shape(ypred)[0],))
        for ind in np.arange(0, np.shape(ypred)[0]):
            maxIndices = np.where(ypred[ind] == np.max(ypred[ind]))[0]
            if (len(maxIndices) > classChunkHalf):
                maxIndex = maxIndices[classChunkHalf]
            else:
                maxIndex = maxIndices[0]
            inverted[ind] = maxIndex + startYear - classChunkHalf

    return inverted


def convert_to_class(data, startYear, classChunkHalf):


    data = np.array(data) - startYear + classChunkHalf
    dataClass = ku.to_categorical(data)

    return dataClass


def create_multiClass(xInput, yOutput, classChunkHalf):



    yMulti = copy.deepcopy(yOutput)

    for stepVal in np.arange(-classChunkHalf, classChunkHalf + 1, 1.):
        if (stepVal == 0):
            continue
        y = yOutput + stepVal

    return xInput, yMulti


def create_multiLabel(yClass, classChunkHalf):


    youtClass = yClass

    for i in np.arange(0, np.shape(yClass)[0]):
        v = yClass[i, :]
        j = np.argmax(v)
        youtClass[i, j - classChunkHalf:j + classChunkHalf + 1] = 1

    return youtClass

### Define useful functions
def invert_year_output(ypred, startYear, classChunk, yearsall):
    inverted_years = convert_fuzzyDecade_toYear(ypred, startYear,
                                                    classChunk, yearsall)

    return inverted_years


def movingAverageInputMaps(data, avgHalfChunk):
    print(np.shape(data))
    dataAvg = np.zeros(data.shape)
    halfChunk = 2

    for iy in np.arange(0, data.shape[1]):
        yRange = np.arange(iy - halfChunk, iy + halfChunk + 1)
        yRange[yRange < 0] = -99
        yRange[yRange >= data.shape[1]] = -99
        yRange = yRange[yRange >= 0]
        dataAvg[:, iy, :, :] = np.nanmean(data[:, yRange, :, :], axis=1)
    return dataAvg




def idf_corrct_prdctn(yrs, ins, model, **params):
    '''
    Function:
    calculating prediction error
        yrs: true years
        prdctn: predicted years
        crrctYr: boolean array 1x#yrs
     '''

    err = yrs[:, 0] - invert_year_output(model.predict(ins),
                                              params['start_year'], params['classChunk'], params['yall'])

    return err
#
def sort_per_ens(errs, indcs, yrs, **params):
    '''
    Function:
    sorting data by ensemble member
        data: data to be sorted (#ens*#yrs x 1)
        indcs: ensemble members
        dtsrt: sorted data (#ens x #yrs x 1)
     '''
    idxy = np.where(np.abs(errs) <= params['bnd'])

    idxy = idxy[0]
    ensYear = np.zeros([len(idxy),2])
    ensYear[:, 0] = yrs[idxy,0]

    k = 0
    l = 0
    for j in range(len(indcs)):
        for i in range(len(params['yearsall'])):
            if np.abs(errs[k]) <= params['bnd']:
                ensYear[l, 1] = indcs[j]
                l += 1
            k += 1

    eYsrt = ensYear[ensYear[:, 0].argsort(),:]
    dtsrt = {}
    numEns = np.zeros([len(params['yall']),2])
    for ys in range(len(params['yall'])):

        indxs = np.where(eYsrt[:,0] == params['yall'][ys])
        numEns[ys, 0] = params['yall'][ys]

        if np.asarray(indxs[0]).any():
            dtsrt[str(params['yall'][ys])] = eYsrt[indxs[0],1]
            numEns[ys,1] = len(indxs[0])
            if params['save']:

                directoryens = params['saveens'] + str(params['yall'][ys]) + '/'

                if os.path.isdir(directoryens):
                    print("Path does exist")
                else:
                    print("Path does not exist:", sys.exc_info()[0])
                    os.mkdir(directoryens)
                np.savez(directoryens + params['filename'], numEns = len(indxs[0]), indxEns = eYsrt[indxs[0],1])


    return dtsrt, numEns, ensYear

def avg_clsstn_prob(ensYear, mxprdt, yrs, indcs, **params):
    '''
    Function:
    sorting data by ensemble member
        data: data to be sorted (#ens*#yrs x 1)
        indcs: ensemble members
        dtsrt: sorted data (#ens x #yrs x 1)
     '''
    yrs =  yrs.reshape(len(indcs), len(params['yearsall']))

    maskd = np.empty(yrs.shape)
    maskd[:,:] = np.nan
    eYsrt = ensYear[ensYear[:, 1].argsort(), :]
    for j in range(len(indcs)):
        crryrs = eYsrt[np.where(eYsrt[:, 1] == indcs[j]),0]
        crryrs = crryrs[0].astype('int')
        for cr in range(len(crryrs)):
            indxs = np.where(yrs[j, :] == crryrs[cr])
            maskd[j,indxs] = 1

    pred_prob = np.nanmean((mxprdt * maskd),axis = 0)


    return pred_prob

def prdct_cmprsn(enstrue, **params):

    a = enstrue[params['pred']]
    if params.get('pred','MLP') == params['net']:
        print(params['net'], 'is correct')

        for key in enstrue.keys():
            if not params['net'] == key:
                b = enstrue[key]
    else:
        print(params['net'], 'is not correct')
        b = enstrue[params['net']]


    for ens in a:
        if (ens == b).sum() == 0:
            return ens
    return print('For %s all ensembles of MLP and CNN where the same' %params['ch_year'])

def select_batch(x: np.ndarray,
                 y: np.ndarray,
                 explanation: np.ndarray,
                 **params):
    """"""
    if params['interpret'] == 'training':
        meanx = np.mean(explanation, axis = 0)
    else:
        meanx = explanation
    nan_flt = np.isnan(meanx).sum(axis = 1).sum(axis = 1)
    indxs = np.where(nan_flt > 0)[0]
    x_out =np.delete(x,indxs, axis=0)
    y_out = np.delete(y, indxs, axis=0)
    exp_out = np.delete(explanation, indxs, axis =1)

    if nan_flt.sum() == 0:
        x_out =x
        y_out = y
        exp_out = explanation 

    return x_out, y_out, exp_out

def select_batch_from_list(x: np.ndarray,
                 y: np.ndarray,
                 explanation: np.ndarray,
                 enssort: dict,
                 da: np.ndarray,) -> list:
    """ Filtering correct explanations by using in dict 
    of correct ensemble members per year

    Args:
        x (np.ndarray): array of inputs (#samples, lat, lon, 1)
        y (np.ndarray): array of outputs (#samples, #categories)
        explanation (np.ndarray): explanation methods (#XAI methods, #samples, lat, lon)
        enssort (dict): keys = years, values = correct ensemble members
        da (np.ndarray): explanation in shape (#XAI methods, #ensembles, #years, lat, lon)

    Returns:
        list: list containing arrays of cleaned inputs, outputs, explanations
    """

    exp = np.empty(da.shape)
    exp[:] = np.nan

    i = 0
    for keys, values in enssort.items():
        if not i:
            strt = int(keys)
        yrs = int(int(keys) - strt)
        vals = np.sort(values, axis=None).astype(int)
        exp[:,vals,yrs,...] = da[:,vals,yrs,...]
        i += 1

    exp = exp.reshape(explanation.shape)
    meanx = exp.mean(axis = 0)
    nan_flt = np.isnan(meanx).sum(axis = 1).sum(axis = 1)
    indxs = np.where(nan_flt > 0)[0]
    x_out =np.delete(x,indxs, axis=0)
    y_out = np.delete(y, indxs, axis=0)
    exp_out = np.delete(explanation, indxs, axis =1)

    if nan_flt.sum() == 0:
        x_out =x
        y_out = y
        exp_out = explanation 

    return x_out, y_out, exp_out

