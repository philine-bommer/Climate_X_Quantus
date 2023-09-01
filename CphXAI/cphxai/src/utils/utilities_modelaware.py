"""
This module includes functions that are useful utilities for interpretation of ANN
using the innvestigate package.

Notes
-----
    Author : Philine Bommer, Dilyara Bareeva
    Based on: https://github.com/albermax/innvestigate/blob/master/examples/notebooks/mnist_compare_methods.ipynb
    Date   : 23 November 2021
    Edit : 23 November 2021

Usage
-----
    [1] getMapPerMethod(model,XXt,YYt,biasBool,annType,classChunk,startYear)

"""
" Import python packages "
import xarray as xr
import numpy as np
from typing import Tuple, Optional, Any
" Import ML libraries "
import tensorflow.compat.v1.keras as keras
# from keras.models import Sequential
import innvestigate
from innvestigate.analyzer import LRP
" Import modules "
from ..utils.utilities_calc import *
import noisegrad_tf.srctf.noisegrad_tf as ng
import noisegrad_tf.srctf.explainers_tf as xg



###############################################################################
###############################################################################
###############################################################################
def LRPcomposite(model_wo_softmax, mode,**kwargs):

    """
    Create analyzer using the LRp composite rule
    :param model_wo_softmax:
    :return:
    """

    analyzer = LRP(model_wo_softmax,
            neuron_selection_mode = mode,
            rule="Z",
            input_layer_rule = "Bounded",
            until_layer_idx = kwargs.get("layer_idx",2),
            until_layer_rule = "Gamma",
            bn_layer_rule = "Z")

    return analyzer

def getMapPerMethodAll(model: keras.Sequential,
               data: np.ndarray,
               XXt: np.ndarray,
               method: Any,
               **params):


    """
    Calculate Explanations without discarding wrong predictions
    """
    print('<<<< Started innvestiage method >>>>')
    annType = params['XAI'].get('annType', 'class')

    ### Reshape into X and T
    if len(data.shape) == 4:
        if params['net'] == 'CNN':
            Xt = XXt.reshape((data.shape[0], data.shape[1],
                              data.shape[2], data.shape[3], 1))
        else:
            Xt = XXt.reshape((data.shape[0], data.shape[1],
                              (data.shape[2] * data.shape[3])))
    elif len(data.shape) == 3:
        if params['net'] == 'CNN':
            Xt = XXt.reshape((data.shape[0],
                              data.shape[1], data.shape[2], 1))
        else:
            Xt = XXt.reshape((data.shape[0],
                              (data.shape[1] * data.shape[2])))

    ### Create the innvestigate analyzer instance for each sample
    if (annType == 'class'):
        model_wo_softmax = innvestigate.utils.model_wo_softmax(model)

        if method[0] == 'LRPcomp':
            analyzer = LRPcomposite(model_wo_softmax,"max_activation", **method[1])

        elif method[1]:
                analyzer = innvestigate.create_analyzer(method[0], model_wo_softmax, **method[1])
        else:
            analyzer = innvestigate.create_analyzer(method[0],model_wo_softmax)

    maps = np.empty(np.shape(Xt))
    maps[:] = np.nan

    if len(data.shape)==4:
        for ens in range(0, np.shape(Xt)[0]):

            for yr in range(0, np.shape(Xt)[1]):
                sample = Xt[ens, yr, ...]
                analyzer_output = analyzer.analyze(sample[np.newaxis, ...])
                maps[ens, yr, ...] = analyzer_output

    elif len(data.shape)==3:

        for yr in range(0, np.shape(Xt)[0]):
            sample = Xt[yr, ...]
            analyzer_output = analyzer.analyze(sample[np.newaxis, ...])
            maps[yr, ...] = analyzer_output


    print('done with innvestigate method')

    return maps

def getMapPerMethod(model: keras.Sequential,
               XXt: np.ndarray,
               Yyt: np.ndarray,
               method: Any,
               **params):


    """
    Calculate Explanations for correctly predicted input samples (precited year <= withinYearInc) and
    averages along the year axis across correctly predicted ensemble members in that year
    :param model: trained network
    :param XXt: input samples that should be explained
    :param Yyt: years of each input sample
    :param method:
    :param params:
    :return: array of explanations (dim. #years in dataset x dim. explanation maps)
    """
    print('<<<< Started innvestiage method >>>>')
    startYear = params.get('startYear', 1920)
    annType = params['XAI'].get('annType', 'class')

    ### Define prediction error
    yearsUnique = np.unique(Yyt)
    percCutoff = 90
    withinYearInc = params['XAI']['yrtol']
    errTolerance = withinYearInc

    if (annType == 'class'):
        err = Yyt[:, 0] - invert_year_output(model.predict(XXt),
                                             startYear, params['classChunk'], params['yall'])

    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Create the innvestigate analyzer instance for each sample
    if (annType == 'class'):
        model_wo_softmax = innvestigate.utils.model_wo_softmax(model)
        if method[0] == 'LRPcomp':
            analyzer = LRPcomposite(model_wo_softmax,"max_activation", **method[1])
        elif method[1]:

            analyzer = innvestigate.create_analyzer(method[0], model_wo_softmax, **method[1])
        else:
            analyzer = innvestigate.create_analyzer(method[0], model_wo_softmax)

    maps = np.empty(np.shape(XXt))
    maps[:] = np.nan
    # analyze each input via the analyzer
    for i in np.arange(0, np.shape(XXt)[0]):

        # ensure error is small, i.e. model was correct
        if (np.abs(err[i]) <= errTolerance):
            sample = XXt[i]
            analyzer_output = analyzer.analyze(sample[np.newaxis, ...])
            maps[i] = analyzer_output

    print('done with explanation')

    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Compute the frequency of data at each point and the average relevance
    ### normalized by the sum over the area and the frequency above the 90th
    ### percentile of the map
    yearsUnique = np.unique(Yyt)
    if params['net'] == 'CNN':
        dTM = maps.reshape((yearsUnique.shape[0],int(np.shape(maps)[0]/yearsUnique.shape[0]),np.shape(maps)[1]*np.shape(maps)[2]))
        deepTaylorMaps = maps.reshape((np.shape(maps)[0],np.shape(maps)[1]*np.shape(maps)[2]))
    else:
        dTM = maps.reshape((yearsUnique.shape[0], int(np.shape(maps)[0] / yearsUnique.shape[0]), np.shape(maps)[1]))
        deepTaylorMaps = maps

    summaryX = np.nanmean(dTM, axis = 1)
    summaryDT = np.zeros((len(yearsUnique), np.shape(deepTaylorMaps)[1]))
    summaryDTFreq = np.zeros((len(yearsUnique), np.shape(deepTaylorMaps)[1]))
    summaryNanCount = np.zeros((len(yearsUnique), 1))

    for i, year in enumerate(yearsUnique):
        ### Years within N years of each year
        j = np.where(np.abs(Yyt - year) <= withinYearInc)[0]

        ### Average relevance across ensembles
        a = np.nanmean(deepTaylorMaps[j, ...], axis=0)
        summaryDT[i, :] = a[np.newaxis, ...]

        ### Frequency of non-nans
        nancount = np.count_nonzero(~np.isnan(deepTaylorMaps[j, 1]))
        summaryNanCount[i] = nancount

        ### Frequency above percentile cutoff
        count = 0
        for k in j:
            b = deepTaylorMaps[k, :]
            if (~np.isnan(b[0])):
                count = count + 1
                pVal = np.percentile(b, percCutoff)
                summaryDTFreq[i, :] = summaryDTFreq[i, :] + np.where(b >= pVal, 1, 0)
        if (count == 0):
            summaryDTFreq[i, :] = 0
        else:
            summaryDTFreq[i, :] = summaryDTFreq[i, :] / count

    print('<<<< Completed cleaning for wrong (> +- %s year) prediction  >>>>' %errTolerance)

    return (summaryX, maps, summaryDTFreq, summaryNanCount)

def NoiseGradMap(model: keras.Sequential,
                 method: Any,
                 XXt: np.ndarray,
                 Yyt: np.ndarray,
                 **params):


    """
    Calculate Explanations
    """
    print('<<<< Started innvestiage method >>>>')
    startYear = params['XAI']['startYear']
    annType = params['XAI']['annType']

    ### Define prediction error
    withinYearInc = params['XAI']['yrtol']
    errTolerance = withinYearInc

    if (annType == 'class'):
        err = Yyt[:, 0] - invert_year_output(model.predict(XXt),
                                             startYear, params['classChunk'], params['yall'])

    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Create the innvestigate analyzer instance for each sample
    YClassMulti, decadeChunks = convert_fuzzyDecade(Yyt,startYear,
                                                            params['classChunk'], params['yall'])

    models1 = copy.deepcopy(model)
    if method[0] == 'NoiseGrad':
        explainer = ng.NoiseGrad(model=models1, std=method[1]['std'], n=20)
    else:
        explainer = ng.NoiseGradPlusPlus(model=models1, std=method[1]['std']/2, sg_std=method[1]['sgd']/2, n=20, m=20)

    maps = explainer.enhance_explanation(inputs=XXt, targets=YClassMulti,
                                              explanation_fn=xg.saliency_explainer, **method[1])


    for i in np.arange(0, np.shape(XXt)[0]):

        # ensure error is small, i.e. model was correct
        if not (np.abs(err[i]) <= errTolerance):

            maps[i,...] = np.nan


    print('done with explanation')

    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Compute the frequency of data at each point and the average relevance
    ### normalized by the sum over the area and the frequency above the 90th
    ### percentile of the map
    yearsUnique = np.unique(Yyt)
    if params['net'] == 'CNN':
        dTM = maps.reshape((yearsUnique.shape[0],int(np.shape(maps)[0]/yearsUnique.shape[0]),np.shape(maps)[1]*np.shape(maps)[2]))
    else:
        dTM = maps.reshape((yearsUnique.shape[0],int(np.shape(maps)[0]/yearsUnique.shape[0]),np.shape(maps)[1]))

    summaryDT = np.nanmean(dTM, axis=1)
    print('<<<< Completed cleaning for wrong (> +- %s year) prediction  >>>>' %errTolerance)

    del models1

    return(summaryDT, maps)

def dataarrayXAI(lats, lons, var, directory, SAMPLEQ, type, method):
    print('\n>>> Using netcdf4LENS function!')

    name = method + '_UAI_YearlyMaps_%s_20ens_T2M_%s_annual.nc' % (SAMPLEQ,type)
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
                                 description='Explanation maps for random sampling of each year',
                                 units='unitless relevance', title='relevance'))

    lrpnc.to_netcdf(filename)

    print('*Completed: Created netCDF4 File!')