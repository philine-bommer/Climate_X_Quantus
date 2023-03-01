"""
Train the model on the different LENS-SINGLE runs to get slopes of observations

Reference  : Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 30 September 2020
"""

### Import packages
" Import python packages "
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pdb
import yaml


" Import modules with utilities "
import cphxai.src.utils.utilities_data as dt
import cphxai.src.utils.utilities_calc as uc


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

cfd = os.path.dirname(os.path.abspath(__file__))
settings = yaml.load(open('%s/Data_config.yaml' %cfd), Loader=yaml.FullLoader)


### Plotting defaults
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries
dirdata = settings['dirhome'] + 'Data/Raw/' # directory of raw data
datasetsingle = settings['datafiles'] # do not change in config
seasons = settings['season'] # do not change in config
timelens = np.arange(settings['start_year'], settings['end_year'] + 1, 1) #do not change in config
yearsall = [timelens]


### Set settings:
bs = settings['bs']
reg_name = settings['reg_name']
variq = settings['variq']
params = settings['params']

# Set paths for in and out-files.
dirhome = settings['dirhome']
avgHalfChunk = settings['aHC']

directorydata = settings['dirhome'] + 'Data/Training/'
if not os.path.isdir(directorydata):
    print("Data path does not exist")
    os.mkdir(directorydata)



sis = settings['sis']
singlesimulation = datasetsingle[sis]

for seas in range(len(seasons)):

    monthlychoice = seasons[seas]
    lat_bounds, lon_bounds = dt.regions(reg_name)

    ### Define primary dataset to use
    dataset = singlesimulation
    modelType = dataset

    ### Whether to test and plot the results using obs data
    test_on_obs = True
    dataset_obs = settings['dataset_obs']
    if dataset_obs == '20CRv3':
        year_obsall = np.arange(yearsall[sis].min(), settings['obs_year'] + 1, 1)
    else:
        year_obs = year_obsall

    obsyearstart = year_obsall.min()

    segment_data_factor = settings['sdf']# Split the data: value of .8 will use 80% training, 20% testing; etc.
    avgHalfChunk = settings['aHC']


    # Parameters for label generation.
    debug = settings['train']['debug']
    classChunkHalf = settings['train']['classChunkHalf']
    classChunk = params['classChunk']
    exp = settings['exp']

    expList = settings['train']['expList']
    expN = np.size(expList)

    params['yearsall'] = yearsall[sis]


    # Load simulation model data.
    lat_bounds, lon_bounds = dt.regions(reg_name)
    data, lats, lons = dt.read_primary_dataset(variq, dataset,
                                                   lat_bounds,
                                                   lon_bounds, monthlychoice, dirdata)
    # Load observation data.
    data_obs, lats_obs, lons_obs = dt.read_obs_dataset(variq, dataset_obs,
                                                           lat_bounds,
                                                           lon_bounds, monthlychoice, yearsall, sis, dirdata)



    # Data segmentation for network training.
    random_segment_seed = settings['rss']  # None

    Xtrain, Ytrain, Xtest, Ytest, Xtest_shape, Xtrain_shape, data_train_shape, \
    data_test_shape, testIndices, trainIndices, random_segment_seed = dt.segment_data(data, segment_data_factor,
                                                                                      sis, yearsall[sis], debug,
                                                                                      random_segment_seed)

    # Convert year into decadal class.
    startYear = yearsall[sis].min()
    endYear = yearsall[sis].max()
    params['startYear'] = startYear
    YtrainClassMulti, decadeChunks = uc.convert_fuzzyDecade(Ytrain,
                                                            startYear,
                                                            classChunk, yearsall[sis])
    YtestClassMulti, __ = uc.convert_fuzzyDecade(Ytest,
                                                 startYear,
                                                 classChunk, yearsall[sis])

    # Reshape data for CNN.
    if params['net'] == 'CNN':

        Xtrain = dt.shape_data(Xtrain, data)
        Xtest = dt.shape_data(Xtest, data)

    # For use later
    XtrainS, XtestS, stdVals = uc.standardize_data(Xtrain, Xtest)
    Xmean, Xstd = stdVals


    # Prepare observational data for model testing.
    dataOBSERVATIONS = data_obs
    latsOBSERVATIONS = lats_obs
    lonsOBSERVATIONS = lons_obs


    Xobs = dataOBSERVATIONS.reshape(dataOBSERVATIONS.shape[0],
                                    dataOBSERVATIONS.shape[1] * dataOBSERVATIONS.shape[2])

    yearsObs = np.arange(dataOBSERVATIONS.shape[0]) + obsyearstart
    if dataset_obs == '20CRv3':
        yearsObs = year_obsall

    # Standardize observation data
    annType = settings['train']['annType']
    years = np.arange(startYear, endYear + 1, 1)
    Xmeanobs = np.nanmean(Xobs, axis=0)
    Xstdobs = np.nanstd(Xobs, axis=0)

    XobsS = (Xobs - Xmeanobs) / Xstdobs
    XobsS[np.isnan(XobsS)] = 0

    if (annType == 'class'):
        if params['net'] == 'CNN':
            XobsS = dt.shape_data(XobsS, dataOBSERVATIONS)


    # Print data variable for reference.
    print('\n\n------------------------')
    print(variq, '= Variable!')
    print(monthlychoice, '= Time!')
    print(reg_name, '= Region!')
    print(lat_bounds, lon_bounds)
    print(dataset, '= Model!')
    print(dataset_obs, '= Observations!\n')

    # Save preprocessed data.
    dataname = 'Preprocessed_data_%s_CESM1_obs_%s.npz' %(params['net'],dataset_obs)
    np.savez(directorydata + dataname, trainModels=trainIndices,
             testModels=testIndices, Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest, Ytest=Ytest,
             Xmean=Xmean, Xstd=Xstd, lats=lats, lons=lons, YtrainClassMulti = YtrainClassMulti,
             testIndices =testIndices, trainIndices = trainIndices,
             YtestClassMulti = YtestClassMulti, decadeChunks = decadeChunks,data = data, data_obs =data_obs, XobsS = XobsS,
             yearsObs = yearsObs, Xstdobs= Xstdobs, Xmeanobs= Xmeanobs,
             lons_obs = lons_obs, lats_obs =lats_obs, obsyearstart =obsyearstart)


    datasetsingle = settings['datafiles']
    seasons = settings['season']

# Save data preperation settings.
settingsout = yaml.load(open('%s/Data_config.yaml' %cfd), Loader=yaml.FullLoader)
settingsout['diroutput'] = directorydata
settingsout['data_name'] = dataname
settingsout['data_raw'] = dirdata

yaml_outfile = '%s/%s_data.yaml' %(cfd, params['net'])

with open(yaml_outfile, 'w') as yaml_file:
    yaml.dump(settingsout, yaml_file, default_flow_style=False)


