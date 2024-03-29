"""
Plot composites under different settings for NoiseGrad

Reference  : Deser et al. [2020, JCLI]
Author    : Philine Bommer
Date      : 11 November 2020
"""
### Import packages
import palettable.cubehelix as cm
import matplotlib as mpl
import numpy as np
import xarray as xr
import yaml
import os
import sys
import pdb
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

## Import self-contributed packages
import cphxai.src.utils.utilities_data as ud
import ccxai.src.visuals.maps as vm
import ccxai.src.utils.utilities_statistics as uss
import ccxai.src.utils.utils_load as ul
import ccxai.src.utils.utils_preprocess as up




### Data preliminaries
cfd = os.path.dirname(os.path.abspath(__file__))
config = yaml.load(open('%s/plot_config.yaml' %cfd), Loader=yaml.FullLoader)
data_settings = yaml.load(open('%s/Data_config.yaml' %cfd), Loader=yaml.FullLoader)
settings = yaml.load(open('%s/%s_results.yaml' %(cfd,data_settings['params']['net'])), Loader=yaml.FullLoader)
other_dt = config['other_dt']
dirdata = settings['diroutput'] + other_dt
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
config['net'] = data_settings['params']['net']

params = config['params']
params['datafiles'] = data_settings['datafiles'][0]
params['dataset'] = data_settings['datafiles']
params['plot']['cmap'] = "coolwarm"
params['plot']['label'] = r'Relevances'
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



'''Find correct and wrong predictions'''
mod = 0
if 'mean' in params['plot']['ens']: 
    params['ens'] = params['plot']['ens']
else:
    directoryM = config['dirquantus'] + 'MLP/' + other_dt + str(mod) +'/'
    directoryC = config['dirquantus'] + 'CNN/' + other_dt + str(mod) +'/'
    directoryensM = directoryM + str(config['ch_yrs'][0]) + '/'
    directoryensC = directoryC + str(config['ch_yrs'][0]) + '/'
    filename = 'Correct_Ensembles_network%s_%s_%s.npz' %(mod,data_settings['params']['interpret'],config['datasets'][0])
    ensMLP = np.load(directoryensM + filename)
    ensCNN = np.load(directoryensC + filename)
    params['pred'] = data_settings['params']['net']
    params['ch_year'] = config['ch_yrs'][0]
    params['net'] = data_settings['params']['net']
    ensss = {'MLP': ensMLP['indxEns'],'CNN': ensCNN['indxEns']}
    params['ens'] = int(ensss[params['net']][0])

xaidataname = settings['dataname']
params['plot']['dir'] = directoryfigure
params['dirdata'] = settings['diroutput'] + other_dt
# config['datatype'] = 'uncleaned'
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

shap_exp=np.load(settings['diroutput'] + other_dt + 'DeepShap_UAI_YearlyMaps_1_20ens_T2M_training_ALL_annual.npz')
if 'mean' in params['plot']['ens']: 
    shap = np.nanmean(shap_exp['values'],axis = 2)
else:
    shap = shap_exp['values'][:,:,params['ens'],:,:,:]
shapx = dts[0,:,:,:,:]
shapx.values = shap[0,...]
dts = xr.concat((dts,shapx.expand_dims({"model":1})),dim = "model")
method_names.append('DeepShap')
methods_name.append('DeepShap')

params_raw = params
params_raw['dirdata'] = data_settings['dirhome'] + 'Data/' + 'Raw/' 
if config['add_raw']:
    raw_data = ul.raw_data(config['variq'], **params_raw)
    if 'mean' in params['plot']['ens']: 
        raw_data = raw_data.mean(dim= 'ensembles', skipna=True)
    else:
        raw_data = raw_data[{'ensembles':params['ens']}]

years = np.arange(config['start_year'], config['end_year'] + 1, 1)
init = (len(years)//num_y)



params['plot']['dir'] = directoryfigure
params['dirdata'] = config['dir_raw']


params['plot']['cmap'] =mpl.cm.get_cmap('coolwarm')
params['nyears'] = num_y


if 'MLP' in config['net']:
    method_names = ['Gradient', 'Smoothgrad', 'InputGradients', 'IntGrad', 'LRPz', 'LRPab', 'NG', 'FG', 'DeepShap']
    methods_name = ['Gradient', 'Smoothgrad', 'InputGradients', 'IntGrad', 'LRPz', 'LRPab', 'NoiseGrad', 'FusionGrad', 'DeepShap']
else:
    method_names = ['Gradient', 'Smoothgrad', 'InputGradients', 'IntGrad', 'LRPz', 'LRPab', 'LRPcomp', 'NG', 'FG', 'DeepShap']
    methods_name = ['Gradient', 'Smoothgrad', 'InputGradients', 'IntGrad', 'LRPz', 'LRPab', 'LRPcomp', 'NoiseGrad', 'FusionGrad', 'DeepShap']


dts = dts.assign_coords({'model':methods_name})
dtss = dts.assign_coords({'years': years})

lrps =up.vis_norm(dtss)


idvdtss =lrps.rename({'model':'models'})
idvdtss = idvdtss[{'samples':mod}]

runs = np.asarray([config['ch_yrs']])

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
params['plot']['colorf'] = ["#008080", "#FFA500", "#124E78", "#d62728", "#8B7D6B","#97FFFF","#BF3EFF"]
region = params['plot']['region'] # for zoomed window in plot_config.yaml set 'plot: region = [270,360,20,80]'
print (f"Set plot region to: {region}")

if params['plot']['region']:
    params['plot']['figname'] = namesfig + '_indvMaps_NA.pdf'
else:
    params['plot']['figname'] = namesfig + '_indvMaps.pdf'

params['plot']['axis'] = [0.85, 0.3, 0.4, 0.03]
params['plot']['figsize']=(20, 13)


'''Several Years throughout century'''
init = (len(years)//num_y)//2
# runs = np.arange(years[0],config['end_year'],30)
runs = np.asarray([config['ch_yrs']])


params['plot']['add_raw'] = config['add_raw']


xaidts = ud.yrs_inDataset(idvdtss.copy(), config['ch_yrs'], runs, dtss['model'].values)

if config['add_raw']:
    raw_data = ud.yrs_inDataset(raw_data.copy().rename({'model': 'models'}),config['ch_yrs'], 
                                   runs, raw_data['model'].values)
    raw_data = raw_data.assign_coords({'models':[ r'$\bar{T}$']})
    plotdata = xr.concat((raw_data, xaidts), 'models')
    
plotdata = plotdata.rename({'time': 'periods'})

if config['net'] == 'MLP':
    plotdata = plotdata.assign_coords({'models': [r'$\bar{T}$', 'gradient', 'SmoothGrad', 'input x \n gradients',
                                                  'Integrated \n Gradients', 'LRPz', r'LRP$\alpha \beta$', 'NoiseGrad',
                                                  'FusionGrad', 'DeepSHAP']})
 
else:
    plotdata = plotdata.assign_coords({'models':[r'$\bar{T}$', 'gradient', 'SmoothGrad', 'input x \n gradients', 
                                                 'Integrated \n Gradients', 'LRPz',r'LRP$\alpha \beta$','LRP \n composite', 
                                                 'NoiseGrad', 'FusionGrad', 'DeepSHAP']})


plotdata =  plotdata.sel(models = [r'$\bar{T}$', 'LRPz','DeepSHAP'])


if params['plot']['region']:
    print (f"Plotting region: {region}")
    plotdata = plotdata.sel(lon = slice(params['plot']['region'][0],params['plot']['region'][1]),
                              lat = slice(params['plot']['region'][2],params['plot']['region'][3]))
vm.plot_PaperMaps(plotdata, **params)