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
import cphxai.src.utils.utilities_calc as uc
import ccxai.src.utils.utils_load as ul

if __name__=='__main__':


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

        # savemod = 'model_%s' %mod
        # model.save(directoryeval + savemod + '.tf')
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


        if params['interpret'] == 'training':
            XtrainS, XtestS, stdVals = uc.standardize_data(Xtrain, Xtest)

            # Reshape Data.
            inpts = np.append(XtrainS, XtestS, axis=0) # Inputs
            indcs = np.append(trainensnum, testensnum, axis=0) # Ensemble indicies for training and testing data.
            otpts = np.append(YtrainClassMulti, YtestClassMulti, axis=0) #Prediction targets (class vector with probabilities)
            outs = np.append(Ytrain, Ytest, axis=0) #Years of the input maps (Regression targets)

            # Identify correct predictions.
            params['yall'] = params['yearsall']
            crrctYrs = uc.idf_corrct_prdctn(outs, inpts, model, **params)

            # Save correct prediction markers (years per ensemble memeber)
            params['filename'] = 'Correct_Ensembles_network%s_%s_%s' %(mod,config['datatype'],config['datasets'][0])
            params['saveens'] = directoryeval
            enssort, ensnums, ensYrs = uc.sort_per_ens(crrctYrs, indcs, outs, **params)
     
        else:

            # Reshape Data.
            inpts = XobsS # Inputs
            indcs = np.append(trainensnum, testensnum, axis=0) # Ensemble indicies for training and testing data.
            otpts, decadeChunks = uc.convert_fuzzyDecade(
                                    yearsObs, params['yearsall'].min(),
                                    params['classChunk'], params['yearsall']) #Prediction targets (class vector with probabilities)
            outs = yearsObs #Years of the input maps (Regression targets)

            # # Identify correct predictions.
            params['yall'] = params['yearsall']
            # crrctYrs = uc.idf_corrct_prdctn(outs, inpts, model, **params)

            # # Save correct prediction markers (years per ensemble memeber)
            # params['filename'] = 'Correct_Ensembles_network%s_%s_%s' %(mod,config['datatype'],config['datasets'][0])
            # params['saveens'] = directoryeval
            # enssort, ensnums, ensYrs = uc.sort_per_ens(crrctYrs, indcs, outs, **params)

        
        # Load individual explanations.
        net_samps = str(settings['SAMPLEQ']) + '_'
        kwargs = {'settings':[net_samps]}
        if settings['params']['interpret'] == 'both':
            if params['interpret'] == 'training':
                xaidataname = [settings['dataname'][0],settings['dataname'][1]]
            else:
                xaidataname = [settings['dataname'][2],settings['dataname'][3]]
        else:
            xaidataname = settings['dataname']

        if config['exptype'] == 'uncleaned':
            filelist = ul.list_subs(directorydataoutput, xaidataname[1],**kwargs)
        else:
            filelist = ul.list_subs(directorydataoutput, xaidataname[0],**kwargs)
            filelist2 = ul.list_subs(directorydataoutput, xaidataname[1],**kwargs)

        directories_data = []
        for i in range(len(filelist)):
            directories_data.append(directorydataoutput)

        method_names = []
        methods_name = []
        for methods in xai_methods:
            method_names.append(methods[2])
            methods_name.append(methods[2])

        method_names[0] = 'Gradient'
        params['order'] = method_names

        filesort = ul.sortfilelist(filelist, **params)
        params['ens'] = mod
        dts = ul.data_concat_ens(filesort, directories_data, 'model', **params)
        explanations = dts.values
        
        if config['exptype'] == 'uncleaned':
            if params['interpret'] == 'training':
                explanations = explanations.reshape(dts.shape[0],dts.shape[1]*dts.shape[2],dts.shape[3],dts.shape[4])
            else:
                explanations = explanations.reshape(dts.shape[0],dts.shape[1],dts.shape[2]*dts.shape[3])
                explanations = explanations[:,:,None,:]

        else:
            if params['interpret'] == 'training':
                explanations = explanations.reshape(dts.shape[0],dts.shape[1]*dts.shape[2],dts.shape[3],dts.shape[4])
            else:
                explanations = explanations.reshape(dts.shape[0],dts.shape[1],dts.shape[2]*dts.shape[3])
                explanations = explanations[:,:,None,:]

        # Sort out incorrect predictions.
        if config['exptype'] == 'cleaned':
            # ipts, otp, exp = uc.select_batch(inpts, otpts, explanations, **params)
            # if exp.size == 0 or exp[-1].sum() == np.nan:
            #     del ipts, otp, exp
            #     filesort2 = ul.sortfilelist(filelist2, **params)
            #     dts2 = ul.data_concat_ens(filesort2, directories_data, 'model', **params)
            #     explanations = dts2.values
            #     if params['interpret'] == 'training':
            #         explanations = explanations.reshape(dts.shape[0],dts.shape[1]*dts.shape[2],dts.shape[3],dts.shape[4])
            #     else:
            #         explanations = explanations.reshape(dts.shape[0],dts.shape[1],dts.shape[2]*dts.shape[3])
            #         explanations = explanations[:,:,None,:]
            #     inpts, otpts, explanations = uc.select_batch_from_list(inpts, otpts, explanations, enssort, dts2.values)
            # else:
            #     inpts, otpts, explanations = uc.select_batch(inpts, otpts, explanations, **params)
            inpts, otpts, explanations = uc.select_batch(inpts, otpts, explanations, **params)


        savename1 = 'Postprocessed_data_%s_%s.npz' %(config['datasets'][0],config['exptype'])
        np.savez(directoryeval + savename1, Input=inpts, Labels = otpts, Batchsize = 32,  wh = [lat, lon],
                 NetworkParams ={'lr':lr_here,'momentum':0.9, 'nesterov':True, 'loss':'binary_crossentropy',
                                 'metrics': 'tf.keras.metrics.categorical_accuracy', 'optimizer': 'SGD'})
        idx = config['masked']
        mask = np.zeros((len(lat),len(lon)))
        mask[idx[0]:idx[1], idx[2]:idx[3]] = np.ones((idx[1]-idx[0], idx[3]-idx[2]))
        maskNan = np.zeros((len(lat), len(lon)))
        maskNan[idx[0]:idx[1], idx[2]:idx[3]] = np.nan
        print('<<<< Created mask at latitude %s - %s, longitude %s - %s  >>>>' % (lat[idx[0]],lat[idx[1]], lon[idx[2]], lon[idx[3]]))

        
        dts = dts.assign_coords({'model': methods_name})

        for n, models in enumerate(dts.model.values):
            savename = 'Explanations_%s_%s_%s' % (config['datasets'][0],models,config['exptype'])
            np.savez(directoryeval + savename + '.npz', Explanation =explanations[n,:,:,:],
                    MaskNorthAtlantik= mask, maskNan= maskNan)

        settingsout = yaml.load(open('%s/Post_config.yaml' %cfd), Loader=yaml.FullLoader)

        settingsout['xai_methods'] = methods
        settingsout['net'] = params['net']
        settingsout['outdata'] = directoryeval + savename1
        settingsout['diroutput'] = directoryeval
        settingsout['outmod'] = settings['dirnet'] + sublisth5[mod]#directoryeval + savemod


        with open('%s/Post_config.yaml' %cfd, 'w') as yaml_file:
            yaml.dump(settingsout, yaml_file, default_flow_style=False)
