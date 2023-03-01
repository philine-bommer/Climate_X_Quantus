"""
Train the model on the different LENS-SINGLE runs to get slopes of observations

Reference  : Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 30 September 2020
"""

### Import packages
" Import python packages "
import os
import sys
import copy
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
import tensorflow as tf
import pandas as pd
import scipy.stats as stats
import warnings
import pdb
import yaml


" Import modules with utilities "
import cphxai.src.utils.utilities_network as net
import cphxai.src.utils.utilities_calc as uc
import cphxai.src.utils.utilities_modelaware as xai_aw


###############################################################################
###############################################################################
###############################################################################

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

### Prevent tensorflow 2.+ deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

cfd = os.path.dirname(os.path.abspath(__file__))
data_settings = yaml.load(open('%s/Data_config.yaml' %cfd), Loader=yaml.FullLoader)
settings = yaml.load(open('%s/%s_config.yaml' %(cfd,data_settings['params']['net'])), Loader=yaml.FullLoader)

data_settings = yaml.load(open('%s/%s_data.yaml' %(cfd,data_settings['params']['net'])), Loader=yaml.FullLoader)

# Set Plotting defaults.
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})


# Check out and in paths or create.
dirhome = settings['dirhome']
directoryNet = dirhome + 'Network/'
if not os.path.isdir(directoryNet):
    print("Network path does not exist")
    os.mkdir(directoryNet)

directorydataoutput = settings['dirhome'] + 'Data/Training/' + data_settings['params']['net'] + '/'
if not os.path.isdir(directorydataoutput):
    print("outpath does not exist")
    os.mkdir(directorydataoutput)

dirfig = dirhome + 'Figures/'
if not os.path.isdir(dirfig):
    print("Figure path does not exist")
    os.mkdir(dirfig)

directoryfigure = dirfig + data_settings['params']['net'] + '/'
if not os.path.isdir(directoryfigure):
    print("Figure path does not exist")
    os.mkdir(directoryfigure)



# Set general settings.
params = settings['params']

# Set data fixed settings.
sis = settings['sis']
singlesimulation = settings['datafiles'][sis]
seasons = settings['season']
timelens = np.arange(settings['start_year'], settings['end_year'] + 1, 1)
yearsall = [timelens]
avgHalfChunk = settings['aHC']
params['yearsall'] = yearsall[sis]

# Set settings for training.
SAMPLEQ = settings['SAMPLEQ']
bs = settings['bs']
lr = settings['lr']
reg_name = settings['reg_name']
useGPU = settings['useGPU']
params['plot'] = settings['plot_in_train']
annType = settings['train']['annType']
params['penalty'] = settings['train']['ridge_penalty']
params['actFun'] = settings['train']['actFun']
params['opt'] = tf.keras.optimizers.SGD(lr=lr,momentum=settings['train']['momentum'],
                                        nesterov=settings['train']['nesterov'])

# Load or create list of random network seeds to ensure reproducability.
try:
    dicts = np.load(dirhome + 'Experiments/Random_Seed_List_nsamp_%s.npz' % SAMPLEQ)
    rand_seed = dicts['random_seed']
except:
    rand_seed = net.generate_random_seed(params)
    svname = 'Experiments/Random_Seed_List_nsamp_%s.npz' % SAMPLEQ
    np.savez(dirhome + svname, random_seed=rand_seed)




# Defining explanation methods with hyperparamter settings.
uai = settings['uai']
params['XAI']['annType'] = annType
params['XAI']['single'] = singlesimulation
methods = [
    ("gradient",              {}, "Gradient"),
    ("smoothgrad",            {"augment_by_n": params['XAI']['augment_by_n'],
                               "noise_scale": params['XAI']['noise_scale']}, "Smoothgrad"),
    ("input_t_gradient",      {}, "InputGradients"),
    ("integrated_gradients",  {}, "IntGrad"),
    ("lrp.z",                 {}, "LRPz"),
    ("lrp.alpha_1_beta_0",                 {}, "LRPab"),
    ("LRPcomp", {"layer_idx": int(params['XAI']['layer_idx'])}, "LRPcomp"),
    ("NoiseGrad", {"sgd": params['XAI']['noise_scale'],
                    "std": params['XAI']['std']}, "NG"),
    ("FusionGrad", {"sgd": params['XAI']['noise_scale'],
                    "std": params['XAI']['std']}, "FG")]

# Delete LRP-composite for the MLP as it is equivalent to LRP-z for networks with only dense layers.
if data_settings['params']['net'] == 'MLP':
    del methods[6]




# Initiate result files.
results_model = []
xaimapstime = {k[0]: [] for k in methods}
xaimapstimeALL = {k[0]: [] for k in methods}


# Define primary dataset.
modelType = singlesimulation


# Setup tensorflow1 session and graph for network training.
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
K.clear_session()

# Set iterable training parameters.
for ih in np.arange(0, len(settings['train']['hiddensList'])):
    hiddens = [settings['train']['hiddensList'][ih]]

startYear = params['yearsall'].min()
endYear = params['yearsall'].max()
params['startYear'] = startYear


# ---------------------------------------------------------------------------

# Load Data from data file.
prep_data= np.load(data_settings['diroutput'] + data_settings['data_name'])
data = prep_data['data']
Xtrain = prep_data['Xtrain']
Ytrain = prep_data['Ytrain']
Xtest = prep_data['Xtest']
Ytest = prep_data['Ytest']
lats = prep_data['lats']
lons = prep_data['lons']
YtrainClassMulti = prep_data['YtrainClassMulti']
YtestClassMulti = prep_data['YtestClassMulti']
decadeChunks = prep_data['decadeChunks']
XobsS = prep_data['XobsS']
yearsObs = prep_data['yearsObs']
Xstdobs = prep_data['Xstdobs']
Xmeanobs = prep_data['Xmeanobs']
testIndices = prep_data['testIndices']
trainIndices = prep_data['trainIndices']
yearsObs = prep_data['yearsObs']
lons_obs = prep_data['lons_obs']
lats_obs = prep_data['lats_obs']
obsyearstart = prep_data['obsyearstart']

# ---------------------------------------------------------------------------

# Standardization of data.
XtrainS, XtestS, stdVals = uc.standardize_data(Xtrain, Xtest)
Xmean, Xstd = stdVals

# Set noise scale acc. to data var. for SmoothGrad, NoiseGrad and FusionGrad.
nsc =np.round(methods[1][1]['noise_scale'] * (np.max(XtrainS) - np.min(XtrainS)),1)
methods[1][1]['noise_scale'] = nsc

for isample in range(SAMPLEQ):

    # ---------------------------------------------------------------------------
    # Set ANN parameters and initialize results file.
    random_network_seed = rand_seed[0,isample]
    params['random_network_seed'] = int(random_network_seed)
    params['hiddens'] = hiddens

    experiment_result = pd.DataFrame(columns=['actual iters', 'hiddens', 'cascade',
                                              'RMSE Train', 'RMSE Test',
                                              'ridge penalty', 'zero mean',
                                              'zero merid mean', 'land only?', 'ocean only?'])
    params['experiment_result'] = experiment_result

    # Create network, train and save results.
        # params['iterations'] is for the # of sample runs the model will use. Must be a
        # list, but can be a list with only one object.
    exp_result, model = net.test_train_loopClass(Xtrain,
                                                 YtrainClassMulti,
                                                 Xtest,
                                                 YtestClassMulti,
                                                 params)

    model.summary()
    results_model.append(np.asarray([seasons, isample, np.asarray(exp_result['Training Results'])]))

    # Save trained model file.
    dirname = directoryNet
    savename = modelType + '_' + params['net'] + '_' + str(isample) + '_'+ settings['variq'] + '_' + str(SAMPLEQ)
    savenameModelTestTrain = modelType + '_' + str(isample) + '_'+ settings['variq'] + '_' + str(SAMPLEQ)

    if (reg_name == 'Globe'):
        regSave = ''
    else:
        regSave = '_' + reg_name

    savename = savename + regSave
    model.save(dirname + savename + '.tf')
    print('saving ' + savename)

    # ---------------------------------------------------------------------------
    # Analyse regression results.
    years = np.arange(startYear, endYear + 1, 1)
    XobsS[np.isnan(XobsS)] = 0

    if (annType == 'class'):

        # Convert to Year.
        YpredObs = uc.convert_fuzzyDecade_toYear(model.predict(XobsS),
                                              startYear,
                                              params['classChunk'], params['yearsall'])
        YpredTrain = uc.convert_fuzzyDecade_toYear(model.predict((Xtrain - Xmean) / Xstd),
                                                startYear,
                                                params['classChunk'], params['yearsall'])
        YpredTest = uc.convert_fuzzyDecade_toYear(model.predict((Xtest - Xmean) / Xstd),
                                               startYear,
                                               params['classChunk'], params['yearsall'])

    # Select observations for regression.
    obsactual = yearsObs
    obspredic = YpredObs

    # Perform Regression.
    slopeobs, interceptobs, r_valueobs, p_valueobs, std_errobs = stats.linregress(obsactual, obspredic)
    slope, intercept, r_value, p_value, std_err = stats.linregress(Ytest[:,0], YpredTest)



    # ---------------------------------------------------------------------------
    # Explain model predictions.
    params['XAI']['startYear'] = startYear
    if uai:
        params['yall'] = params['yearsall']
        params['XAI']['Indices'] = np.append(np.array(testIndices), np.array(trainIndices), axis =0)
        xaimaps = {k[0]: [] for k in methods}
        xaimapsALL = {k[0]: [] for k in methods}
        for method in methods:

            # Run NG and FG.
            if 'NoiseGrad' in method[0] or 'FusionGrad' in method[0]:

                #Define Hyperparameters.
                method[1]['dtype'] = "flat"
                if params['net'] == 'CNN':
                    method[1]['img_height'] = XtrainS.shape[1]
                    method[1]['img_width'] = XtrainS.shape[2]
                method[1]['sgd'] = nsc
                xaimaps[method[0]], xaimapsALL[method[0]] = xai_aw.NoiseGradMap(
                                                            model,method, np.append(XtrainS, XtestS, axis=0),
                                                            np.append(Ytrain, Ytest, axis=0), **params)
                # Reset noise scale for logging (otherwise file error).
                method[1]['sgd'] = params['XAI']['noise_scale']
            else:
                xaimaps[method[0]], xaimapsALL[method[0]], summaryDTFreq, summaryNanCount = xai_aw.getMapPerMethod(
                                                                model,np.append(XtrainS, XtestS, axis=0),
                                                                np.append(Ytrain, Ytest, axis=0), method, **params)



    ## Print Analysis settings and results.
    print('\n\n------------------------')
    print(settings['variq'], '= Variable!')
    print(reg_name, '= Region!')
    print(modelType, '= Model!')
    print(r_valueobs, slopeobs, p_valueobs, '= stats obs')
    print(r_value, slope, p_value, '= stats test')

    # Append explanation maps for the trained model
    for method in methods:
        xaimapstime[method[0]].append(np.array(xaimaps[method[0]]))
        xaimapstimeALL[method[0]].append(np.array(xaimapsALL[method[0]]))

    print('\n\n<<<<<<<<<< COMPLETED ITERATION = %s >>>>>>>>>>>\n\n' % (isample + 1))

for method in methods:

    print(method[2])

    # Reshape to shape #models, ensemble mem. (only for ALL), years, lat, lon.
    xaimapsall_m = np.array(xaimapstime[method[0]])
    xaimapsAll_m = np.array(xaimapstimeALL[method[0]])
    xaimapsall_m = xaimapsall_m.reshape(1, SAMPLEQ,len(params['yearsall']), len(lats), len(lons))
    xaimapsAll_m = xaimapsAll_m.reshape(1, SAMPLEQ, data.shape[0], len(params['yearsall']), len(lats), len(lons))

    #Save.
    xai_aw.dataarrayXAI(lats, lons, xaimapsall_m, directorydataoutput, SAMPLEQ, str(params['interpret'])+'_cleaned', method[2])
    xai_aw.dataarrayXAI(lats, lons, xaimapsAll_m, directorydataoutput, SAMPLEQ, str(params['interpret'])+'_ALL', method[2])


settingsout = yaml.load(open('%s/%s_config.yaml' %(cfd,params['net'])), Loader=yaml.FullLoader)

## Formatting methods file for result tracking:
methods[1][1]['noise_scale'] = params['XAI']['noise_scale']

settingsout['xai_methods'] = methods
settingsout['dataname'] = [str(params['interpret'])+'_cleaned',str(params['interpret'])+'_ALL']
settingsout['diroutput'] = directorydataoutput
settingsout['dirfig'] = directoryfigure
settingsout['dirnet'] = directoryNet

# Log settings for training and explanation.
with open('%s/%s_results.yaml' %(cfd, params['net']), 'w') as yaml_file:
    yaml.dump(settingsout, yaml_file, default_flow_style=False)


