
import numpy as np
import pandas as pd
import yaml
import os
import pdb


import quantus
import ccxai.src.utils.utilities_quantus as qtstf
import ccxai.src.utils.utilities_statistics as qtstats

import tensorflow as tf
import tensorflow.compat.v1.keras as keras
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)



# Load config and settings.
cfd = os.path.dirname(os.path.abspath(__file__))
data_settings = yaml.load(open('%s/Data_config.yaml' %cfd), Loader=yaml.FullLoader)
settings = yaml.load(open('%s/%s_results.yaml' %(cfd,data_settings['params']['net'])), Loader=yaml.FullLoader)
post_settings = yaml.load(open('%s/Post_config.yaml' %cfd), Loader=yaml.FullLoader)
config = yaml.load(open('%s/plot_config.yaml' %cfd), Loader=yaml.FullLoader)

# Set paths.
dirdata = settings['diroutput']

dirout = data_settings['dirhome'] + 'Data/Quantus/'+'Baseline/'
if not os.path.isdir(dirout):
    print("path does not exist")
    os.mkdir(dirout)

# Set general settings.
params = config['params']
config['net'] = data_settings['params']['net']
params['net'] = config['net']
params['interpret'] = config['params']['interpret']
numM = post_settings['mod']
num_y = config['nyears']
n_smpl = settings['params']['SAMPLEQ']
datasetsingle = settings['datafiles']
years = np.arange(config['start_year'], config['end_year'] + 1, 1)
init = (len(years)//num_y)

dirout = dirout + params['interpret'] + '/'
if not os.path.isdir(dirout):
    print("path does not exist")
    os.mkdir(dirout)
dirout = dirout + post_settings['exptype'] + '/' 
if not os.path.isdir(dirout):
    print("path does not exist")
    os.mkdir(dirout)


# Load model and data.
all = np.load(post_settings['outdata'], allow_pickle=True)
x_batch = all["Input"].reshape(all["Input"].shape[0], 1, len(all["wh"][0]), len(all["wh"][1]))
y_batch = np.argmax(all["Labels"], axis=1).flatten()
batch_size = all["Input"].shape[0]


model = load_model(post_settings['outmod'], compile=False)
if config['net'] == 'CNN':
    model.layers[0].name = 'input_layer'
    model.layers[4].name = 'dense_0'
print(f"\n Model architecture: {model.summary()}\n")


# Draw n random indices from batch.
nr_samples_viz = 50 # set to adjust number of explanation samples to use for evaluation.

if params['interpret'] == 'training':
    try:
        dicts = np.load(config['dirquantus'] + 'Random_Seed_List_nsamp_%s_%s.npz' % (nr_samples_viz, post_settings['exptype']))
        rand_seed = dicts['random_seed']
    except:
        sample_indices_viz = np.random.choice(np.arange(0, batch_size - 1), size=nr_samples_viz)
        svname = 'Random_Samples_List_nsamp_%s_%s.npz' %(nr_samples_viz, post_settings['exptype'])
        np.savez(config['dirquantus'] + svname, random_seed=sample_indices_viz)
else:
    try:
        dicts = np.load(config['dirquantus'] + 'Random_Seed_obs_List_nsamp_%s_%s.npz' % (nr_samples_viz, post_settings['exptype']))
        rand_seed = dicts['random_seed']
    except:
        sample_indices_viz = np.random.choice(np.arange(0, batch_size - 1), size=nr_samples_viz)
        svname = 'Random_Samples_obs_List_nsamp_%s_%s.npz' %(nr_samples_viz, post_settings['exptype'])
        np.savez(config['dirquantus'] + svname, random_seed=sample_indices_viz)

x_batch = x_batch[sample_indices_viz]
y_batch = y_batch[sample_indices_viz]

# Set explanation hyperparameters.
xai_methods = settings['xai_methods']
methods_name = []

for methods in xai_methods:
    if "SHAP" in methods[0]:
        argsn = methods[1]
        argsn["samples"] = 1
        argsn["batch_size"] = 64
        as_list = list(methods)
        as_list[1] = argsn
        methods = tuple(as_list)
    if methods[0] == 'NoiseGrad' or methods[0] == 'FusionGrad':
        argsn = methods[1]
        if methods[0] == 'NoiseGrad':
            params['XAI']['addxai'] = ['FG']
        elif methods[0] == 'FusionGrad':
            params['XAI']['additional'] = 0
            argsn['sgd'] = np.round(xai_methods[1][1]['noise_scale'] * (np.max(x_batch) - np.min(x_batch)), 1)
        argsn['dtype'] = "flat"
        if config['net'] == 'CNN':
            argsn['img_height'] = all["Input"].shape[1]
            argsn['img_width'] = all["Input"].shape[2]
        argsn['y_out'] = all["Labels"][sample_indices_viz]
        argsn['std'] = 0.25
        as_list = list(methods)
        as_list[1] = argsn
        methods = tuple(as_list)
    methods_name.append(methods[2])

# Set up Random Baseline.
methods_name.append("Control Var. Random Uniform")
xai_methods.append(("Control Var. Random Uniform", {}, "Control Var. Random Uniform"))

# Load explanations.
explanations = {}
for n, method in enumerate(methods_name):
    if 'Random' in method:
        explanations[method] = np.random.rand(*x_batch.shape)
    else:
        savename = 'Explanations_%s_%s_%s' % (config['datasets'][0], method, post_settings['exptype'])
        explanations_raw = np.load(post_settings['diroutput'] + savename + '.npz', allow_pickle=True)
        if n == 0:
            mask = explanations_raw['MaskNorthAtlantik']

        explanations[method] = explanations_raw['Explanation'][sample_indices_viz].reshape(*x_batch.shape)

s_batch = np.tile(mask, (all["Input"].shape[0], 1, 1))
s_batch = np.expand_dims(s_batch, 1)
s_batch = s_batch[sample_indices_viz]

# Set Quantus argument dict.
params['model_weights'] = model.get_weights()
n_smps = config['n_smps']
arguments = {'model': model,
             'x_batch': x_batch,
             'y_batch': y_batch,
             's_batch': s_batch,
             'net': config['net'],
             'y_out': all["Labels"][sample_indices_viz],
             'n_smps': n_smps,
             'n_sms': config['n_sms'],
             'n_iter': int(n_smps / config['n_sms']),
             "num_cl": all["Labels"][sample_indices_viz].shape[0]
             }

# Set up Quantus metrics dicts.
if config['property'] in "Robustness":

    if config['net'] in 'MLP':
        metrics = {
            "AvgSensitivity": quantus.AvgSensitivity(nr_samples=config['n_sms'],
                                                     lower_bound=0.1,
                                                     norm_numerator=quantus.fro_norm,
                                                     norm_denominator=quantus.fro_norm,
                                                     perturb_func=quantus.gaussian_noise,
                                                     similarity_func=quantus.difference,
                                                     disable_warnings=True, normalise=True),
            "LocalLipschitzEstimate": quantus.LocalLipschitzEstimate(
                nr_samples=config['n_sms'],
                perturb_std=0.1,
                perturb_mean=0,
                norm_numerator=quantus.distance_euclidean,
                norm_denominator=quantus.distance_euclidean,
                perturb_func=quantus.gaussian_noise,
                similarity_func=quantus.lipschitz_constant, normalise=True), }
    else:
        metrics = {
            "LocalLipschitzEstimate": quantus.LocalLipschitzEstimate(
                nr_samples=config['n_sms'],
                perturb_std=0.1,
                perturb_mean=0,
                norm_numerator=quantus.distance_euclidean,
                norm_denominator=quantus.distance_euclidean,
                perturb_func=quantus.gaussian_noise,
                similarity_func=quantus.lipschitz_constant, normalise=True), }

    params['min_norm'] = list(metrics.keys())
    config['property'] = "Robustness_0"

elif config['property'] in "Faithfulness":
    metrics = {
        "FaithfulnessCorrelation": quantus.FaithfulnessCorrelation(
            nr_runs=n_smps,
            subset_size=40,
            perturb_baseline="uniform",
            perturb_func=quantus.baseline_replacement_by_indices,
            similarity_func=quantus.correlation_pearson,
            return_aggregate=False,
            normalise=True, ),
        "ROAD": quantus.ROAD(noise=0.01,
                             normalise=True,
                             perturb_baseline="uniform",
                             perturb_func=quantus.noisy_linear_imputation,
                             percentages=np.linspace(1, 50, n_smps).tolist()), }
    params['min_norm'] = ["ROAD"]
    config['property'] = "Faithfulness_0"
else:
    metrics = {
        "Complexity:Complexity": quantus.Complexity(
            normalise=True,
            disable_warnings=True),
        "Complexity:Sparseness": quantus.Sparseness(
            normalise=True,
            disable_warnings=True),
        "Localisation:TopK": quantus.TopKIntersection(
            normalise=True,
            disable_warnings=True,
            k=(int(0.01 * int(all["wh"][0].shape[0]) * int(all["wh"][1].shape[0])))),
        "Localisation:RRA": quantus.RelevanceRankAccuracy(
            normalise=True,
            disable_warnings=True),
        "Randomisation": quantus.ModelParameterRandomisation(layer_order="bottom_up",
                                                             similarity_func=quantus.correlation_spearman,
                                                             normalise=True),  # }
    }

    if config['net'] in 'MLP':
        metrics["RandomLogit"] = quantus.RandomLogit(
            normalise=True,
            num_classes=20,
            similarity_func=quantus.correlation_spearman, )

    params['min_norm'] = ["Complexity:Complexity", "Randomisation", "RandomLogit"]
    config['property'] = "Complexity_0"

# Intiate intermediate results save.
params['meth_name'] = methods_name
csv_files = 'inter_results_%s_xai_%s_%s' % (config['property'],len(xai_methods), config['net'])
params['dirout'] = dirout
params['csvfile'] = csv_files
params['npzfile'] = 'inter_results_%s_xai_%s_%s.npz' % (config['property'],len(xai_methods), config['net'])
print('>>>>> Run %s analysis and baseline test <<<<<' % config['property'])
# Run Quantus.
results = qtstf.run_quantus(arguments,explanations,metrics,xai_methods, **params)

# Save raw results.
dfs = pd.DataFrame.from_dict(results)
dfs = dfs.reindex(methods_name)
dfs.to_pickle(dirout + csv_files + '.pkl')
np.savez(dirout + 'raw_results_%s_xai_%s_%s_%s.npz'% (config['property'],len(xai_methods), config['net'], post_settings['exptype']), raw = dfs.values, xai = dfs.index.to_list(), properties = dfs.columns.values)

# Set aggregation params.
params['num_xai'] = len(methods_name)


# Aggregation
results_mean, results_sem = qtstats.aggregation_mean_var(metrics,xai_methods, results,**params)

# Store results as pandas.
df2 = pd.DataFrame.from_dict(results_sem)
df = pd.DataFrame.from_dict(results_mean)

# Take absolute scores.
df = df.abs()

# Save SEM.
df2.to_pickle(dirout + 'results_%s_SEM_scores_xai_%s_%s_%s.pkl' % (config['property'],len(xai_methods), config['net'], post_settings['exptype']))
np.savez(dirout + 'results_%s_SEM_scores_xai_%s_%s_%s.npz' % (config['property'],len(xai_methods), config['net'], post_settings['exptype']), sem = df2.values, xai = methods_name, properties = df2.columns.values)

# Save abs. normalized scores.
df.to_pickle(dirout + 'results_%s_abs_agg_scores_xai_%s_%s_%s.pkl' % (config['property'],len(xai_methods), config['net'], post_settings['exptype']))
np.savez(dirout + 'results_%s_abs_agg_scores_xai_%s_%s_%s.npz' % (config['property'],len(xai_methods), config['net'], post_settings['exptype']), mean = df.values, xai = methods_name, properties = df.columns.values)

# Save norm. ranks scores.
# df_normalised_rank = qtstats.significance_ranking(df, df2)
# df_normalised_rank.to_csv(dirout + 'results_%s_ranks_xai_%s_%s.csv' % (config['property'],len(xai_methods), config['net']), index=False, header=False)

# Statistics: brier skill score
bss_mean, bss_sem = qtstats.bss_mean_var(metrics,xai_methods, results,**params)

bss2 = pd.DataFrame.from_dict(bss_sem)
bss = pd.DataFrame.from_dict(bss_mean)

bss = bss.reindex(methods_name)
bss2 = bss2.reindex(methods_name)


# Save SEM BSS.
bss2.to_pickle(dirout + 'bss_%s_SEM_scores_xai_%s_%s_%s.pkl' % (config['property'],len(xai_methods), config['net'], post_settings['exptype']))
np.savez(dirout + 'bss_%s_SEM_scores_xai_%s_%s_%s.npz'% (config['property'],len(xai_methods), config['net'], post_settings['exptype']), sem = bss2.values, xai = methods_name, properties = df2.columns.values)

# Save mean BSS.
bss.to_pickle(dirout + 'bss_%s_abs_agg_scores_xai_%s_%s_%s.pkl' % (config['property'],len(xai_methods), config['net'], post_settings['exptype']))
np.savez(dirout + 'bss_%s_abs_agg_scores_xai_%s_%s_%s.npz' % (config['property'],len(xai_methods), config['net'], post_settings['exptype']), mean = bss.values, xai = methods_name, properties = df.columns.values)


# Log used data.
np.savez(dirout + 'data_%s_%s_%s.npz' % (config['property'],config['net'], post_settings['exptype']),x_batch =x_batch, y_batch= y_batch,
         s_batch= s_batch, net = config['net'], y_out= all["Labels"][sample_indices_viz], n_smps=n_smps,
         n_sms=  config['n_sms'], n_iter=int(n_smps/ config['n_sms']),
         num_cl= all["Labels"][sample_indices_viz].shape[0], reference = dfs.values[-1])

