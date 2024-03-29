"""
@author: Philine L. Bommer
"""

### Import packages
" Import python packages "
import os
import yaml
import sys
import pdb

import numpy as np
import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
import tensorflow.compat.v1.keras as keras



" Import modules with utilities "
import ccxai.src.utils.utils_load as ul
import ccxai.src.visuals.statistics as sts
import cphxai.src.utils.utilities_calc as uc
import ccxai.src.utils.utils_load as ul




cfd = os.path.dirname(os.path.abspath(__file__))
config = yaml.load(open('%s/Post_config.yaml' %cfd), Loader=yaml.FullLoader)
data_settings = yaml.load(open('%s/Data_config.yaml' % cfd), Loader=yaml.FullLoader)
settings = yaml.load(open('%s/%s_results.yaml' % (cfd, data_settings['params']['net'])), Loader=yaml.FullLoader)

data_settings = yaml.load(open('%s/%s_data.yaml' % (cfd, data_settings['params']['net'])), Loader=yaml.FullLoader)
dirhome = data_settings['dirhome']
net = data_settings['params']['net']

params = config['params']
params['interpret'] = data_settings['params']['interpret']
params['net'] = net
directorydataoutput = settings['diroutput']
datasetsingle = config['datafiles']
seasons = config['season']
lr_here = settings['lr']
params['dirdata'] = data_settings['data_raw']
nsamples = settings['SAMPLEQ']
params['seasons'] = settings['season']
params['reg_name'] = settings['reg_name']
params['dataset_obs'] = data_settings['dataset_obs']
params['start_year'] = settings['start_year']
params['end_year'] = settings['end_year']

dircorrect = settings['dirnet'] + 'Ensembles/'
if os.path.isdir(dircorrect):
    print("Path does exist")
else:
    print("Path does not exist:", sys.exc_info()[0])
    os.mkdir(dircorrect)
params['dirpaper'] = dircorrect
xai_methods = settings['xai_methods']

yearsall = []
if config['datatype'] == 'training':
    for i in range(len(config['datasets'])):
        timex = np.arange(config['start_year'], config['end_year'] + 1, 1)
        yearsall.append(timex)

dirhome = data_settings['dirhome']
try:
    dicts = np.load(dirhome + 'Random_Seed_List_nsamp_%s.npz' % config['params']['SAMPLEQ'])
    rand_seed = dicts['random_seed']
    filelist = ul.list_files_from_seeds(settings['dirnet'], rand_seed)
except:
    filelist = ul.list_files(settings['dirnet'])


for ds in range(len(yearsall)):

    params['yearsall'] = yearsall[ds]

    sublisth5, sublistnpz = ul.list_multisubs(filelist, datasetsingle[ds] + '_' + net, 'tf', str(params['rss']))

    mod = config['mod'] # leave at 0 in config if only 1 trained model.
    directoryeval = dirhome + 'Data/' + 'Quantus/' + net + '/'

    if os.path.isdir(directoryeval):
        print("Path does exist")
    else:
        print("Path does not exist:", sys.exc_info()[0])
        os.mkdir(directoryeval)

    directoryeval = directoryeval + str(mod) + '/'

    if os.path.isdir(directoryeval):
        print("Path does exist")
    else:
        print("Path does not exist:", sys.exc_info()[0])
        os.mkdir(directoryeval)


    model = keras.models.load_model(settings['dirnet'] + sublisth5[mod],compile = False)

    model.compile(optimizer=keras.optimizers.SGD(lr=lr_here, momentum=0.9, nesterov=True),
                    loss='binary_crossentropy',
                    metrics=[keras.metrics.categorical_accuracy], )

   
    prep_data = np.load(data_settings['diroutput'] + data_settings['data_name'])
    Xtrain = prep_data['Xtrain']
    Ytrain = prep_data['Ytrain']
    Xtest = prep_data['Xtest']
    Ytest = prep_data['Ytest']
    lat = prep_data['lats']
    lon = prep_data['lons']
    YtrainClassMulti = prep_data['YtrainClassMulti']
    YtestClassMulti = prep_data['YtestClassMulti']
    decadeChunks = prep_data['decadeChunks']
    testensnum  = prep_data['testIndices']
    trainensnum = prep_data['trainIndices']
    XobsS = prep_data['XobsS']
    yearsObs = prep_data['yearsObs']
    Xstdobs = prep_data['Xstdobs']
    Xmeanobs = prep_data['Xmeanobs']
    lons_obs = prep_data['lons_obs']
    lats_obs = prep_data['lats_obs']
    obsyearstart = prep_data['obsyearstart']


    XtrainS, XtestS, stdVals = uc.standardize_data(Xtrain, Xtest)

    # Reshape Data.
    inpts = np.append(XtrainS, XtestS, axis=0) # Inputs
    indcs = np.append(trainensnum, testensnum, axis=0) # Ensemble indicies for training and testing data.
    otpts = np.append(YtrainClassMulti, YtestClassMulti, axis=0) #Prediction targets (class vector with probabilities)
    outs = np.append(Ytrain, Ytest, axis=0) #Years of the input maps (Regression targets)

    params['yall'] = params['yearsall']

    inpts_obs = XobsS # Inputs
    indcs_obs = np.append(trainensnum, testensnum, axis=0) # Ensemble indicies for training and testing data.
    otpts_obs, decadeChunks = uc.convert_fuzzyDecade(
                            yearsObs, params['yearsall'].min(),
                            params['classChunk'], params['yearsall']) #Prediction targets (class vector with probabilities)
    outs_obs = yearsObs #Years of the input maps (Regression targets)

    accuracy = np.mean(np.argmax(YtestClassMulti, axis = 1) == np.argmax(model.predict(XtestS), axis = 1))
    print('accuracy %s = %s' %(net,accuracy))


    classChunk = settings['params']['classChunk']

    YpredObs = uc.convert_fuzzyDecade_toYear(model.predict(XobsS),
                                            params['yearsall'].min(),
                        params['classChunk'], params['yearsall'])
    YpredTrain = uc.convert_fuzzyDecade_toYear(model.predict(XtrainS),
                                                params['yearsall'].min(),
                        params['classChunk'], params['yearsall'])
    YpredTest = uc.convert_fuzzyDecade_toYear(model.predict(XtestS), 
                                            params['yearsall'].min(),
                        params['classChunk'], params['yearsall'])


    variq = settings['variq']


    years = np.arange(config['start_year'], config['end_year'] + 1, 1)

    directoryfigure = data_settings['dirhome'] + 'Figures/' + net + '/'

    if os.path.isdir(directoryfigure):
        print("Path does exist")
    else:
        print("Path does not exist:", sys.exc_info()[0])
        os.mkdir(directoryfigure)


    params = config['params']
    params['plot']['figname'] = 'NetworkPerformance_%s_scatter.pdf' %(net)


    sts.beginFinalPlot(YpredTrain, YpredTest, Ytrain, Ytest, testensnum, years, yearsObs, YpredObs, trainensnum , yearsall, settings['sis'], accuracy, obsyearstart,
                    variq, directoryfigure, params['plot']['figname'], net)