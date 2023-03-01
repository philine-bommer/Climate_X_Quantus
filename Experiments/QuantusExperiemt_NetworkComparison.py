"""
@author: Philine L. Bommer (p.l.bommer@tu-berlin.de)


"""
import numpy as np
import pandas as pd
import yaml
import os
import pdb


# Plotting specifics.
import seaborn as sns
import matplotlib.pyplot as plt


import quantus
import ccxai.src.utils.utilities_quantus as qtstf
import ccxai.src.visuals.statistics as stats
import ccxai.src.utils.utilities_statistics as qtstats


from keras.models import load_model
import tensorflow as tf
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

dirout = data_settings['dirhome'] + 'Evaluation/'
if not os.path.isdir(dirout):
    print("Results path does not exist")
    os.mkdir(dirout)

directoryfigure = config['dirpaper'] + 'Figures/' + 'Quantus/'
if not os.path.isdir(directoryfigure):
    print("Figure path does not exist")
    os.mkdir(directoryfigure)

# Set general settings.
params = config['params']
config['net'] = data_settings['params']['net']
params['net'] = config['net']
numM = post_settings['mod']
num_y = config['nyears']
n_smpl = settings['params']['SAMPLEQ']
datasetsingle = settings['datafiles']
years = np.arange(config['start_year'], config['end_year'] + 1, 1)
init = (len(years)//num_y)
config['net'] = data_settings['params']['net']


# Load model and data.
all = np.load(post_settings['outdata'], allow_pickle=True)
x_batch = all["Input"].reshape(all["Input"].shape[0], 1, len(all["wh"][0]), len(all["wh"][1]))
y_batch = np.argmax(all["Labels"], axis=1).flatten()
batch_size = all["Input"].shape[0]

model = load_model(post_settings['outmod']+'.tf', compile=False)
if config['net'] == 'CNN':
    model.layers[0].name = 'input_layer'
    model.layers[4].name = 'dense_0'
print(f"\n Model architecture: {model.summary()}\n")


# Draw n random indices from batch.
nr_samples_viz = 50
sample_indices_viz = np.random.choice(np.arange(0, batch_size-1), size=nr_samples_viz)

x_batch = x_batch[sample_indices_viz]
y_batch = y_batch[sample_indices_viz]


# Set explanation hyperparameters.
xai_methods = settings['xai_methods']
methods_name = []

for methods in xai_methods:
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
if config['base']:
    methods_name.append("Control Var. Random Uniform")
    xai_methods.append(("Control Var. Random Uniform", {}, "Control Var. Random Uniform"))

# Load explanations.
explanations = {}
for n, method in enumerate(methods_name):
    if 'Random' in method:
        explanations[method] = np.random.rand(*x_batch.shape)
    else:
        savename = 'Explanations_%s_%s' % (config['datasets'][0], method)
        explanations_raw = np.load(post_settings['diroutput'] + savename + '.npz' , allow_pickle=True)
        if n == 0:
            mask = explanations_raw['MaskNorthAtlantik']

        explanations[method] = explanations_raw['Explanation'][sample_indices_viz]

s_batch = np.tile(mask, (all["Input"].shape[0], 1, 1))
s_batch = np.expand_dims(s_batch, 1)
s_batch = s_batch[sample_indices_viz]


# Set Quantus argument dict.
n_smps = config['n_smps']
params['sample_index'] = sample_indices_viz
arguments = {'model': model,
                'x_batch': x_batch,
                'y_batch': y_batch,
                's_batch': s_batch,
                'net': config['net'],
                'y_out': all["Labels"][sample_indices_viz],
                'n_smps': n_smps,
                'n_sms': config['n_sms'],
                'n_iter': int(n_smps/ config['n_sms']),
                "num_cl": all["Labels"][sample_indices_viz].shape[0]
}


# Plotting configs.
spyfig ='analysis_networks_spyder_plot_xai_%s_%s.pdf' %(len(xai_methods),config['net'])
colours_order = ["#008080",
                 "#FFA500",
                 '#329932',
                  "#d62728",
                 "#97FFFF",
                 "#BF3EFF",
                 "#8A360F",
                '#ff6961',
                '#6a3d9a',
                '#e31a1c',
                '#ff7f00',
                '#cab2d6',
                '#b15928',
                '#67001f',
                '#d6604d',
                '#92c5de',
                '#4393c3',
                '#053061']

methods_order = methods_name



# Set up Quantus metrics dicts.
metrics = {
            "Robustness": quantus.LocalLipschitzEstimate(
                                            nr_samples=config['n_sms'],
                                            perturb_std=0.1,
                                            perturb_mean=0,
                                            norm_numerator=quantus.distance_euclidean,
                                            norm_denominator=quantus.distance_euclidean,
                                            perturb_func=quantus.gaussian_noise,
                                            similarity_func=quantus.lipschitz_constant, normalise=True),
            "Faithfulness": quantus.FaithfulnessCorrelation(
                                            nr_runs=n_smps,
                                            subset_size=40,
                                            perturb_baseline= "uniform",
                                            perturb_func= quantus.baseline_replacement_by_indices,
                                            similarity_func=quantus.correlation_pearson,
                                            return_aggregate= False,
                                            normalise=True,),
            "Localisation": quantus.RelevanceRankAccuracy(
                                                    normalise=True,
                                                    disable_warnings=True),
            "Complexity": quantus.Sparseness(normalise=True,
                                            disable_warnings=True),
            "Randomisation": quantus.ModelParameterRandomisation(layer_order="bottom_up",
                                                   similarity_func=quantus.correlation_spearman,
                                                   normalise=True),}




csv_files = 'analysis_all_properties_data_xai_%s_%s.csv' % (len(xai_methods), config['net'])
params['dirout'] = dirout
params['csvfile'] = csv_files
results = qtstf.run_quantus(arguments,explanations,metrics,xai_methods, **params)


csv_files = 'analysis_raw_scores_data_xai_%s_%s.csv' % (len(xai_methods), config['net'])
dfs = pd.DataFrame.from_dict(results)
dfs.to_csv(dirout + csv_files, index=False, header=False)


# Set aggregation params.
params['n_sms'] = config['n_sms']
params['num_xai'] = len(methods_name)
params['min_norm'] = list(metrics.keys())

# Aggregation
params['min_norm'] = ["Randomisation", "Robustness"]
results_mean, results_sem = qtstats.aggregation_mean_var(metrics,xai_methods, results,**params)

# Store results as pandas.
df2 = pd.DataFrame.from_dict(results_sem)
df = pd.DataFrame.from_dict(results_mean)
# Take absolute scores.
df = df.abs()

# Save SEM.
df2.to_csv(dirout + 'results_SEM_scores_xai_%s_%s.csv'% (len(xai_methods), config['net']), index=False, header=False)

# Save abs. normalized scores.
df.to_csv(dirout + 'results_abs_agg_scors_xai_%s_%s.csv' % (len(xai_methods), config['net']), index=False, header=False)

# Save norm. ranks scores.
df_normalised_rank = qtstats.significance_ranking(df, df2)
df_normalised_rank.to_csv(dirout + 'results_ranks_xai_%s_%s.csv' % (len(xai_methods), config['net']), index=False, header=False)


# Plotting configs.
sns.set(font_scale=3)
plt.style.use('seaborn-white')
plt.rcParams['ytick.labelleft'] = True
plt.rcParams['xtick.labelbottom'] = True

include_titles = config['include_titles']
include_legend = config['include_legend']

# Make spyder graph!

if config['real']:
    # Plot real scores.
    data = [df.columns.values, (df.to_numpy())]
else:
    #plot ranks.
    data = [df_normalised_rank.columns.values, (df_normalised_rank.to_numpy())]

theta = stats.radar_factory(len(data[0]), frame='polygon')
spoke_labels = data.pop(0)


fig, ax = plt.subplots(figsize=(16, 10), subplot_kw=dict(projection='radar'))
fig.subplots_adjust(top=0.85, bottom=0.05)
# pdb.set_trace()
for i, (d, method) in enumerate(zip(data[0], xai_methods)):
    if "Random" in method[2]:
        line = ax.plot(theta, d, label=method[2], color='b', linewidth=5.0, linestyle = '-')
        ax.fill(theta, d, alpha=0.15)
    elif "LRPcomp" in method[2]:
        line = ax.plot(theta, d, label=method[2], color= '#fb9a99', linewidth=3.)
    else:
        if config['net'] == 'CNN' and i >5:
            i -=1
        line = ax.plot(theta, d, label=method[2], color=colours_order[i%len(colours_order)], linewidth=3.)
        ax.fill(theta, d, alpha=0.15)



# Set lables.
if include_titles:
    # Adjust to chosen essential properties for your explanation method -> if neglecting Localisation or Randomisation
    ax.set_varlabels(labels=['Robustness', ' \n\nFaithfulness', '\nLocalisation', '\nComplexity', ' \n\nRandomisation'])
else:
    ax.set_varlabels(labels=[])

if config['real']:
    # Set real scores.
    ax.set_rgrids(np.arange(0, df.values.max() + 0.1, 0.1), labels=[])
else:
    # Set rank scores.
    ax.set_rgrids(np.arange(0, df_normalised_rank.values.max() + 0.5), labels=[])


# Put a legend to the right of the current axis.
if include_legend:
    ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.savefig(directoryfigure + spyfig, dpi=400)