"""
Plot composites (yearly) under different settings for UAI Plus

Reference  : Deser et al. [2020, JCLI]
Author    : Philine Bommer
Date      : 11 November 2020
"""
'''Python Packages'''
import palettable.cubehelix as cm
import matplotlib as mpl
import numpy as np
import xarray as xr
import yaml
import os
import sys

'''Self-written Packages'''
import cphxai.src.utils.utilities_calc as uc
import ccxai.src.visuals.maps as vm
import ccxai.src.utils.utils_load as ul
import ccxai.src.utils.utils_save as us
import ccxai.src.utils.utilities_statistics as uss
import ccxai.src.utils.utils_preprocess as up




### Data preliminaries
cfd = os.path.dirname(os.path.abspath(__file__))
config = yaml.load(open('%s/plot_config.yaml' %cfd), Loader=yaml.FullLoader)
data_settings = yaml.load(open('%s/Data_config.yaml' %cfd), Loader=yaml.FullLoader)
settings = yaml.load(open('%s/%s_results.yaml' %(cfd,config['net'])), Loader=yaml.FullLoader)
dirdata = settings['diroutput']
num_y = config['nyears']
n_smpl = settings['params']['SAMPLEQ']
datasetsingle = settings['datafiles']
xai_methods = settings['xai_methods']

years = np.arange(config['start_year'], config['end_year'] + 1, 1)
init = (len(years)//num_y)


confp = settings
dirm = confp['dirnet']
seasons = confp['season']
lr_here = settings['lr']
nsamples = confp['params']['SAMPLEQ']
xai_methods = settings['xai_methods']

params = config['params']
params['datafiles'] = data_settings['datafiles'][0]
params['dataset'] = data_settings['datafiles']
params['plot']['cmap'] = "coolwarm"
params['plot']['label'] = r'\textbf{Relevances}'
params['seasons'] = config['seasons']
params['reg_name'] = config['reg_name']
params['dataset_obs'] = data_settings['dataset_obs']
params['start_year'] = settings['start_year']
params['end_year'] = settings['end_year']
params['plot']['models'] = xai_methods
params['plot']['num_model'] = len(xai_methods)

directoryfigure = data_settings['dirhome'] + 'Figures/'

if os.path.isdir(directoryfigure):
    print("Path does exist")
else:
    print("Path does not exist:", sys.exc_info()[0])
    os.mkdir(directoryfigure)

method_names = []
methods_name = []
for methods in xai_methods:
    method_names.append(methods[2])
    methods_name.append(methods[2])


if params['XAI']['additional']:

    for i in range(len(params['XAI']['addxai'])):
        method_names.append(params['XAI']['addType'][i])
        methods_name.append(params['XAI']['addxai'][i])


'''Find correct and wrong predictions'''
mod = 0
directoryM = config['dirquantus'] + 'MLP/' + str(mod) +'/'
directoryC = config['dirquantus'] + 'CNN/' + str(mod) +'/'
directoryensM = directoryM + str(config['ch_yrs'][0]) + '/'
directoryensC = directoryC + str(config['ch_yrs'][0]) + '/'
filename = 'Correct_Ensembles_network%s_%s_%s.npz' %(mod,data_settings['params']['interpret'],config['datasets'][0])
ensMLP = np.load(directoryensM + filename)
ensCNN = np.load(directoryensC + filename)
params['pred'] = config['net']
params['ch_year'] = config['ch_yrs'][0]
params['net'] = config['net']
ensss = {'MLP': ensMLP['indxEns'],'CNN': ensCNN['indxEns']}
params['ens'] = int(ensss[params['net']][0])


xaidataname = settings['dataname']
params['plot']['dir'] = directoryfigure
params['dirdata'] = settings['diroutput']
if config['datatype'] == 'uncleaned':
    filelist = ul.list_subs(dirdata, xaidataname[1])
    namesfig = config['fname'] % (config['net'], xaidataname[1], len(config['ch_yrs']),len(method_names))
else:
    filelist = ul.list_subs(dirdata, xaidataname[0])
    namesfig = config['fname'] % (config['net'], xaidataname[0], len(config['ch_yrs']),len(method_names))
    namesfig = namesfig + '_ens_%s' %(params['ens'])
directories_data = []
for i in range(len(filelist)):
    directories_data.append(dirdata)

params['order'] = method_names

filesort = ul.sortfilelist(filelist, ** params)

dts = ul.data_concatenate(filesort, directories_data, 'model', **params)

shap_exp=np.load(settings['diroutput'] + 'DeepShap_UAI_YearlyMaps_1_20ens_T2M_training_ALL_annual.npz')
shap = shap_exp['values'][:,:,params['ens'],:,:,:]
shapx = dts[0,:,:,:,:]
shapx.values = shap[0,...]
dts = xr.concat((dts,shapx.expand_dims({"model":1})),dim = "model")
method_names.append('DeepShap')
methods_name.append('DeepShap')

if config['add_raw']:
    raw_data = ul.raw_data(config['variq'], **params)
    raw_data = raw_data[{'ensembles':params['ens']}]

years = np.arange(config['start_year'], config['end_year'] + 1, 1)
init = (len(years)//num_y)



params['plot']['dir'] = directoryfigure
params['dirdata'] = config['dir_raw']


params['plot']['cmap'] =mpl.cm.get_cmap('coolwarm')
params['nyears'] = num_y
# methods_name = methods_name[:len(dts.model.values)]
if 'MLP' in config['net']:
    method_names = ['Gradient', 'Smoothgrad', 'InputGradients', 'IntGrad', 'LRPz', 'LRPab', 'NG', 'FG', 'DeepShap']
    methods_name = ['Gradient', 'Smoothgrad', 'InputGradients', 'IntGrad', 'LRPz', 'LRPab', 'NoiseGrad', 'FusionGrad', 'DeepShap']
else:
    method_names = ['Gradient', 'Smoothgrad', 'InputGradients', 'IntGrad', 'LRPz', 'LRPab', 'LRPcomp', 'NG', 'FG', 'DeepShap']
    methods_name = ['Gradient', 'Smoothgrad', 'InputGradients', 'IntGrad', 'LRPz', 'LRPab', 'LRPcomp', 'NoiseGrad', 'FusionGrad', 'DeepShap']


dts = dts.assign_coords({'model':methods_name})
dtss = dts.assign_coords({'years': years})

'''Plot indv. MC samples to see deviations'''

lrps =up.vis_norm(dtss)


idvdtss =lrps.rename({'model':'models'})
idvdtss = idvdtss[{'samples':mod}]


runs = np.asarray([config['ch_yrs']])
ridx = np.where(np.in1d(years,runs))[0]

params['plot']['add_raw'] = 1
params['add_raw'] = 1


plottypes = params['plot']['types']
params['plot']['types'] = 'continuous'
params['plot']['LRPlim'] = 0.0
params['plot']['wd_ratio'] = np.asarray([4.5,4.5,1])
params['plot']['cmaps'] = ['coolwarm','Reds','Reds','Reds',]
params['plot']['cmaps'] = ['coolwarm','Reds','Reds','Reds',]
params['plot']['Eaxis'] = [0.55, 0.05, 0.25, 0.03]
params['plot']['raxis'] = [0.15, 0.05, 0.15, 0.03]
params['plot']['labellist'] = ['K','Relevances']
params['plot']['figsize']=(14, 24)
params['plot']['colorf'] = ["#008080", "#FFA500", "#124E78", "#d62728", "#8B7D6B","#97FFFF","#BF3EFF"]
params['plot']['figname'] = namesfig + '_temporalAverage.pdf'

params['nyears'] = 40
uai = uss.timeperiod_calc(idvdtss, **params)


if config['add_raw']:
    raw_data = uss.timeperiod_calc(raw_data.rename({'model': 'models'}), **params)
    raw_data = raw_data.assign_coords({'models':[ r'$\bar{T}$']})
    plotdata = xr.concat((raw_data, uai), 'models')



if config['net'] == 'MLP':

    plotdata = plotdata.assign_coords({'models': [r'$\bar{T}$', 'gradient', 'SmoothGrad', 'input x \n gradients',
                                                  'integrated \n gradients', 'LRPz', r'LRP$\alpha \beta$', 'NoiseGrad',
                                                  'FusionGrad', 'DeepSHAP']})

else:
    plotdata = plotdata.assign_coords({'models':[r'$\bar{T}$', 'gradient', 'SmoothGrad', 'input x \n gradients', 'integrated \n gradients', 'LRPz',r'LRP$\alpha \beta$','LRP \n composite', 'NoiseGrad', 'FusionGrad', 'DeepSHAP']})


params['plot']['region'] = []
vm.plot_PaperMaps(plotdata, **params)