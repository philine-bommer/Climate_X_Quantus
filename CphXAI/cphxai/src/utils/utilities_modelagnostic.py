"""
This module includes functions that are useful untilities for interpretation of ANN

Notes
-----
    Author : Philine Bommer
    Date   : 27.05.22

Usage
-----
    [1] SHAP(model,XXt,YYt,**params)


"""
" Import python packages "
import time
import numpy as np


" Import ML libraries "
import keras.backend as K
import keras
# import tensorflow.compat.v1.keras.backend as K
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

# import shap

" Import modules "
from ..utils.utilities_calc import *
from ..utils.utilities_data import *

# def SHAP(model: keras.Sequential,
#          XXt: np.ndarray,
#          YYt: np.ndarray,
#          **params):
#     '''
#     Function for the application of gradient-based shapley values[1]
#     [1] https://shap-lrjball.readthedocs.io/en/latest/example_notebooks/gradient_explainer/Explain%20an%20Intermediate%20Layer%20of%20VGG16%20on%20ImageNet.html
#     :param model: trained network subject to explanation
#     :param XXt: input instances subject to explanation
#     :param YYt: prediction/targets for explanation
#     :param params:
#     :return: Tuple [average explanations of correct predictions, explanations]
#     '''
#
#     print('<<<< Started SHAP >>>>')
#     startYear = params.get('startYear', 2080)
#     annType = params.get('annType', 'class')
#
#     ### Define prediction error
#     withinYearInc = params['XAI']['yrtol']
#     errTolerance = withinYearInc
#
#     if (annType == 'class'):
#         err = YYt[:, 0] - invert_year_output(model.predict(XXt),
#                                              startYear, params['classChunk'], params['yall'])
#
#     if params['net'] == 'CNN':
#         Xt = XXt
#
#         exp = shap.GradientExplainer((model.layers[0].input, model.layers[-1].output), Xt.copy())
#         K.get_session().run(tf.initialize_all_variables())
#         shap_values, indexes = exp.shap_values(Xt, ranked_outputs=1)
#
#
#     else:
#         if len(XXt.shape) == 3:
#             Xt = XXt.reshape((XXt.shape[0], XXt.shape[1]*XXt.shape[2]))
#         else:
#             Xt = XXt
#
#         exp = shap.GradientExplainer((model.layers[0].input, model.layers[-1].output), Xt.copy())
#         K.get_session().run(tf.initialize_all_variables())
#         shap_values, indexes = exp.shap_values(Xt, ranked_outputs=1)
#
#
#     # analyze each input via the analyzer
#     shap_values = np.array(shap_values[0])
#     for i in np.arange(0, np.shape(XXt)[0]):
#
#         # ensure error is small, i.e. model was correct
#         if not (np.abs(err[i]) <= errTolerance):
#
#             shap_values[i,...] = np.nan
#
#
#     yearsUnique = np.unique(YYt)
#
#
#     if params['net'] == 'CNN':
#         cleanSHAP = np.nanmean(shap_values.reshape(
#             (int(np.shape(shap_values)[0] / len(yearsUnique)), len(yearsUnique),
#              np.shape(shap_values)[1] * np.shape(shap_values)[2])), axis=0)
#         shap_values = shap_values.reshape((int(np.shape(shap_values)[0]/yearsUnique.shape[0]),yearsUnique.shape[0],np.shape(shap_values)[1]*np.shape(shap_values)[2]))
#     else:
#         cleanSHAP = np.nanmean(shap_values.reshape(
#             (int(np.shape(shap_values)[0] / len(yearsUnique)), len(yearsUnique), np.shape(shap_values)[1])), axis=0)
#         shap_values = shap_values.reshape((int(np.shape(shap_values)[0]/yearsUnique.shape[0]),yearsUnique.shape[0],np.shape(shap_values)[1]))
#
#     print('<<<< Completed cleaning for wrong (> +- %s year) prediction  >>>>' %errTolerance)
#
#
#     return cleanSHAP, shap_values