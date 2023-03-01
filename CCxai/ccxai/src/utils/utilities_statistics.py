from typing import Dict, Any, Tuple
import numpy as np
import copy
import pdb
import math
import pandas as pd
import scipy
from sklearn import metrics
import numpy as np
import xarray as xr
from quantus.helpers.utils import *
from quantus.helpers.normalise_func import *


def correlation_pearson(a: np.array, b: np.array, **kwargs) -> float:
    """Calculate Pearson correlation of two images (or explanations)."""
    return scipy.stats.pearsonr(a, b)[0]

def area_score(results: Any,
                     ** kwargs):
    """
    Implements an area under the curve metric for ROAD graph
    """
    y = np.zeros((len(results.values()),))
    x = np.zeros((len(results.values()),))
    i = 0
    for keys, vals in results.items():

        y[i] = vals
        x[i] = float(keys)

        i+=1
    score =  metrics.auc(x,y)
    return score

def aggregated_score(results: Dict) -> Tuple:
    """
    Implements mean, std distance to 100% accuracy reference for ROAD
    :param results: dictionary with keys - percentage of perturbed pixels, vals: accuracy across perturbed samples
    :return: Tuple of mean and std relevance scores
    """
    ref = np.ones((len(results.values()),))
    valss = np.zeros((len(results.values()),))
    i = 0
    for keys, vals in results.items():
        valss[i] = vals
        i += 1


    dist = ref - valss

    return [np.mean(dist), np.std(dist)]

def timeperiod_calc(
        data: xr.DataArray,
        ** params):
    """
    Calculates temporal average over years in a period of years = (start_year - end_year)//step
    :param data: DataArray with dimensions: [#datasets, #years, lat, lon]
    :param params: Dict with average settings
        params['start_year'] start year
        params['end_year'] end year
        params['nyears'] - number of periods to seperate time range into
    :return: DataArray with #years-dimension replaced by periods
    """
    startY = params['start_year']
    endY = params['end_year']
    step = params['nyears']
    years = np.arange(startY, endY + 1, 1)

    period = []

    for count, i in enumerate(range(0, len(years), step)):
        if count < len(years) // step:
            year_range = str(startY + (count * step)) + '-' + str(startY + ((count + 1) * step))
            period.append(year_range)

            if count == 0:
                comp = data[:, i:i + step, :, :].mean(axis=1)
            else:
                comp = xr.concat((comp, data[:, i:i + step, :, :].mean(axis=1)), 'periods')

    data_period = comp.assign_coords({'periods':period})

    data_period = data_period.transpose('models','periods','lat','lon')
    return data_period


def significance_ranking(
        data_mean: pd.DataFrame,
        data_var: pd.DataFrame):

    '''
    determine ranking according to score similarity within variation (SEM, std or similar) magnitude
    :param data_mean:
    :param data_var:
    :return:
    '''
    for keys1, vals in data_mean.items():

        for keys2, val in data_mean[keys1].items():
            var = data_var[keys1][keys2]
            mean = val

            var_mag = int(math.floor(math.log10(var)))
            mean_mag = int(math.floor(math.log10(mean)))
            if mean_mag == var_mag:
                data_mean[keys1][keys2] = np.round(mean, abs(mean_mag))
            else:

                data_mean[keys1][keys2] = np.around(mean, abs(var_mag))

    ranks = data_mean.rank(method='max')
    return ranks


def aggregation_mean_var(metrics: Dict,
                         methods: Dict,
                         results: Dict,
                         **params):
    """
    Calculates score satistics including mean and SEM across samples in scores[metric][method]
    :param metrics: dict of quantus metrics
    :param methods: dict of explanation methods
    :param scores:  dict of scores for each metric and each XAI method
    :param params:  kwargs with number of XAI methods and names of the properties (network comparison see defaults)
                or metrics that underlie normalization according to Eq.
    :return:
    """
    # Set params.
    num_xai = params.get('num_xai', 8)
    string_list = params.get('min_norm', ["Randomisation", "Robustness"])

    # Initialize result dicts.
    means = {}
    var = {}
    # Aggregate mean and SEM.
    for metric, metric_func in metrics.items():
        means[metric] = {}
        var[metric] = {}
        unnormed_scores = []
        for methoddict in methods:
            method = methoddict[2]
            if metric is "ROAD":
                u_sc = []
                for r in range(len(results[metric][method])):
                    agg_score = area_score(results[metric][method][r])
                    u_sc.append(agg_score)
                unnormed_scores.append(np.array(u_sc))
            elif type(results[metric][method]) is dict:
                u_scores = []
                for vals in results[metric][method].values():
                    u_scores.append(vals)

                unnormed_scores.append(np.array(u_scores).flatten())
            else:
                unnormed_scores.append(np.array(results[metric][method]).flatten())

        unnormed_scores = np.array(unnormed_scores)
        unnormed_scores = np.abs(unnormed_scores)
        if metric in string_list:
            min_score = np.min(unnormed_scores, axis=0)
            min_scores = np.repeat(min_score[np.newaxis, :], num_xai, axis=0)
            scores = min_scores / unnormed_scores
        else:
            max_score = np.max(unnormed_scores, axis=0)
            max_scores = np.repeat(max_score[np.newaxis, :], num_xai, axis=0)
            scores = unnormed_scores / max_score

        for i, methoddict in enumerate(methods):
            meth = methoddict[2]
            means[metric][meth] = np.mean(scores[i, :])
            var[metric][meth] = np.std(scores[i, :]) / np.sqrt(scores.shape[1])

    return means, var


