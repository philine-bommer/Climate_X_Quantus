### Plotting functions
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import cmocean
import cartopy.crs as ccrs
import pdb

from ..utils.utilities_data import *
from ..utils.utilities_calc import *


def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.xaxis.set_ticks([])


def plot_prediction(Ttest, test_output, Ttest_obs, obs_output, test_on_obs):
    ### Predictions

    plt.figure(figsize=(16, 4))
    plt.subplot(1, 2, 1)
    plt.title('Predicted vs Actual Year for Testing')
    plt.xlabel('Actual Year')
    plt.ylabel('Predicted Year')
    plt.plot(Ttest, test_output, 'o', color='black', label='GCM')

    if test_on_obs == True:
        plt.plot(Ttest_obs, obs_output, 'o', color='deepskyblue', label='obs')
    a = min(min(Ttest), min(test_output))
    b = max(max(Ttest), max(test_output))
    plt.plot((a, b), (a, b), '-', lw=3, alpha=0.7, color='gray')
    # plt.axis('square')
    plt.xlim(a * .995, b * 1.005)
    plt.ylim(a * .995, b * 1.005)
    plt.legend()
    plt.show()


def plot_training_error(nnet):
    ### Training error (nnet)

    plt.subplot(1, 2, 2)
    plt.plot(nnet.getErrors(), color='black')
    plt.title('Training Error per Itobstion')
    plt.xlabel('Training Itobstion')
    plt.ylabel('Training Error')
    plt.show()


def plot_rmse(train_output, Ttrain, test_output, Ttest, data_train_shape, data_test_shape, yearsall, sis, test_on_obs):
    ### rmse (train_output, Ttrain, test_output, Ttest, data_train_shape, data_test_shape)

    Xtrain_shape = train_output.shape
    Xtest_shape = test_output.shape

    plt.figure(figsize=(16, 4))
    plt.subplot(1, 2, 1)
    rmse_by_year_train = np.sqrt(np.mean(((train_output - Ttrain) ** 2).reshape(Xtrain_shape),
                                         axis=0))
    xs_train = (np.arange(data_train_shape) + yearsall[sis].min())
    rmse_by_year_test = np.sqrt(np.mean(((test_output - Ttest) ** 2).reshape(Xtest_shape),
                                        axis=0))
    xs_test = (np.arange(data_test_shape) + yearsall[sis].min())
    plt.title('RMSE by year')
    plt.xlabel('year')
    plt.ylabel('error')
    plt.plot(xs_train, rmse_by_year_train, label='training error',
             color='gold', linewidth=1.5)
    plt.plot(xs_test, rmse_by_year_test, labe='test error',
             color='forestgreen', linewidth=0.7)
    plt.legend()

    if test_on_obs == True:
        plt.subplot(1, 2, 2)
        error_by_year_test_obs = obs_output - Ttest_obs
        plt.plot(Ttest_obs, error_by_year_test_obs, label='obs error',
                 color='deepskyblue', linewidth=2.)
        plt.title('Error by year for obs')
        plt.xlabel('year')
        plt.ylabel('error')
        plt.legend()
        plt.plot((1979, 2020), (0, 0), color='gray', linewidth=2.)
        plt.xlim(1979, 2020)
    plt.show()


def plot_weights(nnet, lats, lons, basemap, hiddens, cascade, Xtrain):
    # plot maps of the NN weights
    plt.figure(figsize=(16, 6))
    ploti = 0
    nUnitsFirstLayer = nnet.layers[0].nUnits

    for i in range(nUnitsFirstLayer):
        ploti += 1
        plt.subplot(np.ceil(nUnitsFirstLayer / 3), 3, ploti)
        maxWeightMag = nnet.layers[0].W[1:, i].abs().max().item()
        drawOnGlobe(((nnet.layers[0].W[1:, i]).cpu().data.numpy()).reshape(len(lats),
                                                                              len(lons)),
                       lats, lons, basemap, vmin=-maxWeightMag, vmax=maxWeightMag,
                       cmap=cmocean.cm.balance)
        if (hiddens[0] == 0):
            plt.title('Linear Weights')
        else:
            plt.title('First Layer, Unit {}'.format(i + 1))

    if (cascade is True and hiddens[0] != 0):
        plt.figure(figsize=(16, 6))
        ploti += 1
        plt.subplot(np.ceil(nUnitsFirstLayer / 3), 3, ploti)
        maxWeightMag = nnet.layers[-1].W[1:Xtrain.shape[1] + 1, 0].abs().max().item()
        drawOnGlobe(((nnet.layers[-1].W[1:Xtrain.shape[1] + 1, 0]).cpu().data.numpy()).reshape(len(lats),
                                                                                                  len(
                                                                                                      lons)),
                       lats, lons, basemap, vmin=-maxWeightMag,
                       vmax=maxWeightMag, cmap=cmocean.cm.balance)
        plt.title('Linear Weights')
    plt.tight_layout()


def plot_results(nnet, train_output, test_output, obs_output, Ttest,
                 Ttrain, Xtrain_shape, Xtest_shape, data_train_shape,
                 data_test_shape, Ttest_obs, lats, lons, basemap, **num):
    ### Calls all our plot functions together
    plots = num.get('plots', 4)

    if plots >= 1:
        plot_prediction(Ttest, test_output, Ttest_obs, obs_output)
    if plots >= 2:
        plot_training_error(nnet)
        plot_rmse(train_output, Ttrain, test_output, Ttest, data_train_shape, data_test_shape)
    if plots == 4:
        plot_weights(nnet, lats, lons, basemap)
    plt.show()


def plot_classifier_output(class_prob, test_class_prob, Xtest_shape, Xtrain_shape, yearsall, sis):
    prob = class_prob[-1].reshape(Xtrain_shape)

    plt.figure(figsize=(14, 6))
    plt.plot((np.arange(Xtest_shape[1]) + yearsall[sis].min()),
             prob[:, :, 1].T, '-', alpha=.7)
    plt.plot((np.arange(Xtest_shape[1]) + yearsall[sis].min()),
             (np.mean(prob[:, :, 1], axis=0).reshape(180, -1)),
             'b-', linewidth=3.5, alpha=.5, label='ensemble avobsge')
    plt.title('Classifier Output by Ensemble using Training Data')
    plt.xlabel('year')
    plt.yticks((0, 1), ['Pre-Baseline', 'Post-Baseline'])
    plt.legend()
    plt.show()

    tprob = test_class_prob[0].reshape(Xtest_shape)

    plt.figure(figsize=(14, 6))
    plt.plot(((np.arange(Xtest_shape[1]) + yearsall[sis].min())), tprob[:, :, 1].T, '-',
             alpha=.7)
    plt.plot((np.arange(Xtest_shape[1]) + yearsall[sis].min()),
             (np.mean(tprob[:, :, 1], axis=0).reshape(180, -1)),
             'r-', linewidth=4, alpha=.5, label='ensemble avobsge')
    plt.title('Classifier Output by Ensemble using Test Data')
    plt.xlabel('year')
    plt.yticks((0, 1), ['Pre-Baseline', 'Post-Baseline'])
    plt.legend()
    plt.show()


def beginFinalPlot(YpredTrain, YpredTest, Ytrain, Ytest, testIndices, years, yearsObs, YpredObs,
                   rm_ensemble_mean, trainIndices, yearsall, sis, singlesimulation, obsyearstart,
                   variq, monthlychoice, land_only, ocean_only,directoryfigure, savename, modelType):
    """
    Plot prediction of year
    """

    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})

    fig = plt.figure()
    ax = plt.subplot(111)

    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params('both', length=4, width=2, which='major', color='dimgrey')

    train_output_rs = YpredTrain.reshape(len(trainIndices),
                                         len(years))
    test_output_rs = YpredTest.reshape(len(testIndices),
                                       len(years))

    xs_test = (np.arange(np.shape(test_output_rs)[1]) + yearsall[sis].min())


    for i in range(0, train_output_rs.shape[0]):
        if i == train_output_rs.shape[0] - 1:
            p3 = plt.plot(xs_test, train_output_rs[i, :], 'o',
                          markersize=4, color='lightgray', clip_on=False,
                          alpha=0.4, markeredgecolor='k', markeredgewidth=0.4,
                          label=r'\textbf{%s - Training Data}' % singlesimulation)
        else:
            p3 = plt.plot(xs_test, train_output_rs[i, :], 'o',
                          markersize=4, color='lightgray', clip_on=False,
                          alpha=0.4, markeredgecolor='k', markeredgewidth=0.4)
    for i in range(0, test_output_rs.shape[0]):
        if i == test_output_rs.shape[0] - 1:
            p4 = plt.plot(xs_test, test_output_rs[i, :], 'o',
                          markersize=4, color='crimson', clip_on=False, alpha=0.3,
                          markeredgecolor='crimson', markeredgewidth=0.4,
                          label=r'\textbf{%s - Testing Data}' % singlesimulation)
        else:
            p4 = plt.plot(xs_test, test_output_rs[i, :], 'o',
                          markersize=4, color='crimson', clip_on=False, alpha=0.3,
                          markeredgecolor='crimson', markeredgewidth=0.4)

    if rm_ensemble_mean == False:
        iy = np.where(yearsObs >= obsyearstart)[0]
        linereg = stats.linregress(yearsObs[iy], YpredObs[iy])
        ylineobs = linereg.slope * yearsObs[iy] + linereg.intercept
        plt.plot(yearsObs[iy], YpredObs[iy], 'x', color='deepskyblue',
                 label=r'\textbf{Reanalysis}', clip_on=False)
        plt.plot(yearsObs[iy], ylineobs, '-',
                 color='blue', linewidth=2, clip_on=False)

    plt.xlabel(r'\textbf{ACTUAL YEAR}', fontsize=10, color='dimgrey')
    plt.ylabel(r'\textbf{PREDICTED YEAR}', fontsize=10, color='dimgrey')
    plt.plot(np.arange(yearsall[sis].min(), yearsall[sis].max() + 1, 1),
             np.arange(yearsall[sis].min(), yearsall[sis].max() + 1, 1), '-',
             color='black', linewidth=2, clip_on=False)

    plt.xticks(np.arange(yearsall[sis].min(), 2101, 20), map(str, np.arange(yearsall[sis].min(), 2101, 20)),
               size=6)
    plt.yticks(np.arange(yearsall[sis].min(), 2101, 20), map(str, np.arange(yearsall[sis].min(), 2101, 20)),
               size=6)
    plt.xlim([yearsall[sis].min(), yearsall[sis].max()])
    plt.ylim([yearsall[sis].min(), yearsall[sis].max()])

    plt.title(r'\textbf{[ %s ] $\bf{\longrightarrow}$ RMSE Train = %s; RMSE Test = %s}' % (
    variq, np.round(rmse(YpredTrain[:, ],
                             Ytrain[:, 0]), 1), np.round(rmse(YpredTest[:, ],
                                                                  Ytest[:, 0]),
                                                         decimals=1)),
              color='k',
              fontsize=15)

    iyears = np.where(Ytest < 2000)[0]
    plt.text(yearsall[sis].max(), yearsall[sis].min() + 5,
             r'\textbf{Test RMSE before 2000 = %s}' % (np.round(rmse(YpredTest[iyears,],
                                                                         Ytest[iyears, 0]),
                                                                decimals=1)),
             fontsize=5, ha='right')

    iyears = np.where(Ytest >= 2000)[0]
    plt.text(yearsall[sis].max(), yearsall[sis].min(),
             r'\textbf{Test RMSE after 2000 = %s}' % (np.round(rmse(YpredTest[iyears,],
                                                                        Ytest[iyears, 0]),
                                                               decimals=1)),
             fontsize=5, ha='right')

    leg = plt.legend(shadow=False, fontsize=7, loc='upper left',
                     bbox_to_anchor=(-0.01, 1), fancybox=True, ncol=1, frameon=False,
                     handlelength=1, handletextpad=0.5)
    savefigName = modelType + '_' + variq + '_scatterPred_' + savename
    # plt.annotate(savename,(0,.98),xycoords='figure fraction',
    #              fontsize=5,
    #              color='gray')
    plt.savefig(
        directoryfigure + savefigName + '_%s_land%s_ocean%s.png' % (monthlychoice, land_only, ocean_only),
        dpi=300)
    print(np.round(np.corrcoef(yearsObs, YpredObs)[0, 1], 2))
    return


def plot_average_maps(
        uai: xr.DataArray,
        years: np.ndarray,
        runs: np.ndarray,
        ** params):

    limit = params['plot']['limit']
    # barlim = np.linspace(uai.values.min(), uai.values.max(), 7)#params['plot']['barlim']
    cmap = params['plot']['cmap']
    label = params['plot']['label']
    ens = params['plot']['ens']
    datasetsingleq = params['plot']['models']
    types = params['plot']['types']
    directoryfigure = params['plot']['dir']

    ### Plot variable data for trends
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
    plt.rc('savefig', facecolor='white')
    plt.rc('axes', edgecolor='darkgrey')
    plt.rc('xtick', color='darkgrey')
    plt.rc('ytick', color='darkgrey')
    plt.rc('axes', labelcolor='darkgrey')
    plt.rc('axes', facecolor='black')

    fig, axes = plt.subplots(nrows=params['num_model'], ncols=len(runs), figsize=(10, 6),
                             subplot_kw={'projection': ccrs.Mollweide(central_longitude=0,
                                                                      globe=None, false_easting=None,
                                                                      false_northing=None)},
                             gridspec_kw={"width_ratios": np.repeat(np.array([10]), len(runs)),
                                          "height_ratios": np.repeat(np.array([3]), params['num_model'])})
                             # gridspec_kw={"wspace": 0.35, "hspace": 0.28})
    m = ccrs.PlateCarree()
    r = 0
    for mod in range(uai.values.shape[0]):
        for rs in range(len(runs)):
            ax = axes[mod,rs]

            ax.coastlines(resolution='110m', color='dimgrey', linewidth=0.35)
            ax.set_global()
            # vp = uai[mod,ens, runs[rs], :, :].plot(ax=ax, transform=m, vmax=0.751, vmin=-0.75, cmap=cmap,
            #                       add_colorbar=False, add_labels=False)
            vp = uai[mod, ens, runs[rs], :, :].plot(ax=ax, transform=m, cmap=cmap,
                                                    add_colorbar=False, add_labels=False)

            if any([r == 0, r == 4, r == 8]):
                ax.annotate(r'\textbf{%s}' % datasetsingleq[r], xy=(0, 0), xytext=(-0.1, 0.5),
                             textcoords='axes fraction', color='black', fontsize=10,
                             rotation=90, ha='center', va='center')
            if any([r == 0, r == 1, r == 2, r == 3]):
                ax.annotate(r'\textbf{%s}' % years[runs[r]], xy=(0, 0), xytext=(0.5, 1.22),
                             textcoords='axes fraction', color='darkgrey', fontsize=10,
                             rotation=0, ha='center', va='center')
            r += 1
            # ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.87,0.97),
            #               textcoords='axes fraction',color='k',fontsize=6,
            #               rotation=330,ha='center',va='center')

    ###########################################################################
    cbar_ax = fig.add_axes([0.32, 0.095, 0.4, 0.03])
    cbar = fig.colorbar(vp, cax=cbar_ax, orientation='horizontal')#,
                        #extend='max', extendfrac=0.07, drawedges=False)

    cbar.set_label(label, fontsize=12, color='black', labelpad=1.4)

    # cbar.set_ticks(barlim)
    # cbar.set_ticklabels(list(map(str, barlim)))
    cbar.ax.tick_params(axis='x', size=.01, labelsize=5)
    cbar.outline.set_edgecolor('darkgrey')

    # plt.tight_layout()
    plt.subplots_adjust(top=0.85, wspace=0.01, hspace=0, bottom=0.14)

    plt.savefig(directoryfigure + 'UAI_typ_%s_T2M_allModels.png' % (types), dpi=600)

    return


def plot_xrMaps(
        data: xr.DataArray,
        **params):
    lev = params['plot']['limit']
    level = params['plot']['barlim']
    label = params['plot']['label']

    ### Plot variable data for trends
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
    plt.rc('savefig', facecolor='white')
    plt.rc('axes', edgecolor='darkgrey')
    # plt.rc('xtick', color='darkgrey')
    # plt.rc('ytick', color='darkgrey')
    plt.rc('xtick', color='black')
    plt.rc('ytick', color='black')
    plt.rc('axes', labelcolor='darkgrey')
    plt.rc('axes', facecolor='white')

    yearperiod = data.periods.values
    datasets = data.models.values

    fig1, axes = plt.subplots(nrows=data.shape[0], ncols=data.shape[1], figsize=(17, 8.2),
                                  subplot_kw={'projection': ccrs.Mollweide(central_longitude=0)},)
    m = ccrs.PlateCarree()
    c = 0
    # level = np.asarray(np.linspace(data.min().values, data.max().values, 8))
    # lev = np.around(level, 1).astype(str)

    for mod in range(data.shape[0]):

        uai = data.loc[data.models.values[mod]]
        for yp in range(data.shape[1]):
            ax = axes[mod, yp]

            ax.coastlines(resolution='110m', color='dimgrey', linewidth=0.35)
            ax.set_global()

            cmp = params['plot']['cmap']

            vp = uai[yp, :, :].plot(ax=ax, transform=m, cmap=cmp,
                                            add_colorbar=False, add_labels=False)

            # if any([c == 0, c == 4, c == 8, c == 12]):
            if (c % axes.shape[1]) == 0:
                ax.annotate(r'\textbf{%s}' % datasets[mod], xy=(0, 0), xytext=(-0.1, 0.5),
                            textcoords='axes fraction', color='black', fontsize=10,
                            rotation=90, ha='center', va='center')

            if c < len(yearperiod):
                ax.annotate(r'\textbf{%s}' % yearperiod[yp], xy=(0, 0), xytext=(0.5, 1.22),
                            textcoords='axes fraction', color='darkgrey', fontsize=10,
                            rotation=0, ha='center', va='center')


            c += 1

    ###########################################################################

    cbar_ax = fig1.add_axes([0.32, 0.095, 0.4, 0.03])
    cbar = fig1.colorbar(vp, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(label, fontsize=12, color='black', labelpad=1.4)

    cbar.ax.tick_params(axis='x', size=.02, labelsize=10)
    cbar.outline.set_edgecolor('darkgrey')
    # cbar.set_clim(level.min(),level.max())
    cbar.set_ticks(level)
    cbar.set_ticklabels(lev)

    fig1.subplots_adjust(top=0.85, wspace=0.01, hspace=0, bottom=0.14)

    fig1.savefig(params['plot']['dir'] + params['plot']['figname'], dpi=600)
    # fig1.savefig(params['plot']['dir'] + params['plot']['figname'], dpi=600, bbox_inches='tight')

    plt.close(fig1)

    return


def plot_UAIplus(
        uAIplus: xr.DataArray,
        years: np.ndarray,
        datasets: List,
        ** params):

    lev = params['plot']['limit']
    step = params['nyears']
    level = params['plot']['barlim']
    cmap = params['plot']['cmap']
    label = params['plot']['label']
    ens = params['plot']['ens']
    directoryfigure = params['plot']['dir']

    ### Plot variable data for trends
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
    plt.rc('savefig', facecolor='white')
    plt.rc('axes', edgecolor='darkgrey')
    plt.rc('xtick', color='darkgrey')
    plt.rc('ytick', color='darkgrey')
    plt.rc('axes', labelcolor='darkgrey')
    plt.rc('axes', facecolor='white')

    # uaip = timeperiod_calc(uAIplus, **params)
    uaip = uAIplus
    yearperiod = uAIplus['time'].values

    fig, axes = plt.subplots(nrows=uAIplus.shape[0], ncols=len(years) // step, figsize=(17, 8.2),
                               subplot_kw={'projection': ccrs.Mollweide(central_longitude=0)},
                               )

    m = ccrs.PlateCarree()
    limits = np.empty((uAIplus.shape[0],2))
    c = 0
    datasets = uAIplus.models.values
    for mod in range(uAIplus.shape[0]):
        uai = uaip[mod]
        limits[mod,0] = uai.min().values
        limits[mod,1] = uai.max().values
        for yp in range(len(yearperiod)):

            ax = axes[mod,yp]

            ax.coastlines(resolution='110m', color='dimgrey', linewidth=0.35)
            ax.set_global()

            cmap = params['plot']['cmap']

            if params['plot']['ensmean']:
                if len(uai.shape) > 3:
                    uaim = uai.mean(axis = 0)
                else:
                    uaim = uai

                vp = uaim[yp, :, :].plot.pcolormesh(ax=ax, transform=m, cmap=cmap, levels=level,
                                      add_colorbar=False, add_labels=False)


            else:
                vp = uai[ens, yp, :, :].plot(ax=ax, transform=m, cmap=cmap,
                                      add_colorbar=False, add_labels=False)
            if any([c == 0, c == 4, c == 8, c==12]):
                ax.annotate(r'\textbf{%s}' % datasets[mod], xy=(0, 0), xytext=(-0.1, 0.5),
                             textcoords='axes fraction', color='black', fontsize=10,
                             rotation=90, ha='center', va='center')

            if c < len(yearperiod):#(len(years)//step):
                ax.annotate(r'\textbf{%s}' % yearperiod[yp], xy=(0, 0), xytext=(0.5, 1.22),
                             textcoords='axes fraction', color='darkgrey', fontsize=10,
                             rotation=0, ha='center', va='center')

            c += 1


    ###########################################################################

    cbar_ax = fig.add_axes([0.32, 0.095, 0.4, 0.03])
    cbar = fig.colorbar(vp, cax=cbar_ax, orientation='horizontal')

    cbar.set_label(label, fontsize=12, color='black', labelpad=1.4)


    cbar.set_ticks(level)
    cbar.set_ticklabels(lev)

    cbar.ax.tick_params(axis='x', size=.02, labelsize=10)
    cbar.outline.set_edgecolor('darkgrey')


    plt.subplots_adjust(top=0.85, wspace=0.01, hspace=0, bottom=0.14)

    plt.savefig(directoryfigure + params['plot']['figname'], dpi=600)
    # plt.savefig(directoryfigure + params['plot']['figname'], dpi=600, bbox_inches='tight')#+ 'DiscreteXAImaps_%s_%s_T2M_eps_%s.png' %(params['plot']['time'],params['plot']['types'],params['eps']), dpi=600)

    plt.close(fig)
    return


def plot_XandR(
        data: xr.DataArray,
        **params):
    lev = params['plot']['limit']
    level = params['plot']['barlim']
    label = params['plot']['label']
    ls = params['plot']['labelsize']
    fs = params['plot']['fontsize']

    ### Plot variable data for trends
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
    plt.rc('savefig', facecolor='white')
    plt.rc('axes', edgecolor='darkgrey')
    # plt.rc('xtick', color='darkgrey')
    # plt.rc('ytick', color='darkgrey')
    plt.rc('xtick', color='black')
    plt.rc('ytick', color='black')
    plt.rc('axes', labelcolor='darkgrey')
    plt.rc('axes', facecolor='white')

    yearperiod = data.periods.values
    datasets = data.models.values


    # fig1, axes = plt.subplots(nrows=data.shape[0]+1, ncols=data.shape[1], figsize=(17, 10),
    #                           subplot_kw={'projection': ccrs.Mollweide(central_longitude=0)},
    #                           )
    fig1, axes = plt.subplots(nrows=data.shape[0] + 1, ncols=data.shape[1], figsize=params['plot']['figsize'],
                              subplot_kw={'projection': ccrs.Robinson(central_longitude=0)},
                              gridspec_kw={"hspace": 0.1})


    m = ccrs.PlateCarree()
    c = 0
    for mod in range(data.shape[0]+1):
        if params['plot']['add_raw'] and params['plot']['types'] == 'continuous':
            expl = data[0:data.shape[0]-1,:,:,:]
            if expl.min().values <0:
                xmax = np.max(np.array([np.abs(expl.min().values), expl.max().values]))
                xmin = -xmax
            else:
                xmax = expl.max().values
                xmin = expl.min().values
        if mod == data.shape[0]:
            uai = data.loc[data.models.values[mod-1]]
        else:
            uai = data.loc[data.models.values[mod]]
        for yp in range(data.shape[1]):
            ax = axes[mod, yp]

            if mod < (data.shape[0] - 1):
                cmp = "cool"
            else:
                cmp = params['plot']['cmap']


            if mod == data.shape[0]-1:
                axes[mod, yp].axis('off')
                if yp == (axes.shape[1] - 1):
                    # vp1.set_clim(np.min(level1), np.max(level1))
            # if mod == data.shape[0]-2 and yp == (axes.shape[1]-1):
                    cbar_ax = fig1.add_axes(params['plot']['axis'])
                    cbar1 = fig1.colorbar(vp1, cax=cbar_ax, orientation='horizontal')
                    # cbar1 = fig1.colorbar(vp, ax=axes[mod, :], location='bottom')
                    cbar1.ax.tick_params(axis='x', size=.02, labelsize=ls)
                    cbar1.outline.set_edgecolor('darkgrey')
                    cbar1.set_ticks(params['plot']['barlim'])
                    cbar1.set_ticklabels(params['plot']['limit'])
                    cbar1.set_label(label, fontsize=fs, color='black', labelpad=1.4)
            else:
                ax.coastlines(resolution='110m', color='dimgrey', linewidth=0.35)
                ax.set_global()
                if params['plot']['types'] == 'continuous':
                    # xmax = np.max(np.abs(uai.min().values), uai.max().values)
                    if mod <= data.shape[0] - 2:
                        vp = uai[yp, :, :].plot(ax=ax, transform=m, cmap=cmp, vmin = xmin, vmax = xmax,
                                                add_colorbar=False, add_labels=False)
                    else:
                        vp = uai[yp, :, :].plot(ax=ax, transform=m, cmap=cmp, vmin=uai.min().values, vmax=uai.max().values,
                                                add_colorbar=False, add_labels=False)
                    if mod == (data.shape[0] - 2) and yp == 0:
                        uaids = data.loc[data.models.values[0:mod]]
                        level1 = np.asarray(np.arange(uaids.min().values, uaids.max().values, 0.2))
                        lev1 = np.around(level1, 1).astype(str)

                        del uaids
                else:
                    if len(uai.shape) > 3:
                        uaim = uai.mean(axis=0)
                    else:
                        uaim = uai
                    # xmax = np.max(np.abs(uaim.min().values), uaim.max().values)
                    if mod <= data.shape[0]-2:
                        xmin = 0.
                        xmax = 1.
                        vp = uaim[yp, :, :].plot.pcolormesh(ax=ax, transform=m, cmap=cmp, vmin = xmin, vmax = xmax, levels=level,
                                                        add_colorbar=False, add_labels=False)
                    else:
                        vp = uaim[yp, :, :].plot(ax=ax, transform=m, cmap=cmp, vmin=uaim.min().values, vmax=uaim.max().values,
                                                add_colorbar=False, add_labels=False)


                    lev1 = params['plot']['limit']
                    level1 = params['plot']['barlim']

                if mod <= (data.shape[0] - 2):
                    # pdb.set_trace()
                    vp1 = vp




            if (c % axes.shape[1]) == 0:
                if mod == data.shape[0]:
                    ax.annotate(r'\textbf{%s}' % datasets[mod-1], xy=(0, 0), xytext=(-0.1, 0.5),
                                textcoords='axes fraction', color='black', fontsize=fs,
                                rotation=90, ha='center', va='center')
                elif mod == (data.shape[0]-1):
                    ax.annotate('', xy=(0, 0), xytext=(-0.1, 0.5),
                                textcoords='axes fraction', color='black', fontsize=fs,
                                rotation=90, ha='center', va='center')
                else:
                    ax.annotate(r'\textbf{%s}' % datasets[mod], xy=(0, 0), xytext=(-0.1, 0.5),
                            textcoords='axes fraction', color='black', fontsize=fs,
                            rotation=90, ha='center', va='center')

            if c < len(yearperiod):
                ax.annotate(r'\textbf{%s}' % yearperiod[yp], xy=(0, 0), xytext=(0.5, 1.22),
                            textcoords='axes fraction', color='darkgrey', fontsize=fs,
                            rotation=0, ha='center', va='center')



            c += 1

    ###########################################################################
    level = np.asarray(np.linspace(uai.min().values, uai.max().values, 8))
    lev = np.around(level, 1).astype(str)
    cbar = fig1.colorbar(vp, ax=ax,  orientation='vertical')
    cbar.set_label(r'\textbf{stdized. T}', fontsize=ls, color='black', labelpad=1.4)

        # cbar_ax = fig1.add_axes([0.32, 0.095, 0.4, 0.03])
        # cbar = fig1.colorbar(vp, cax=cbar_ax, orientation='horizontal')
        # cbar.set_label(label, fontsize=12, color='black', labelpad=1.4)

    cbar.ax.tick_params(axis='x', size=.02, labelsize=fs)
    cbar.outline.set_edgecolor('darkgrey')
    cbar.set_ticks(level)
    cbar.set_ticklabels(lev)

    fig1.subplots_adjust(top=0.85, wspace=0.01, hspace=0, bottom=0.14)

    # fig1.savefig(params['plot']['dir'] + params['plot']['figname'], dpi=600)
    fig1.savefig(params['plot']['dir'] + params['plot']['figname'], dpi=600, bbox_inches='tight')

    plt.close(fig1)

    return

def plot_XAIcomparison(
        data: xr.DataArray,
        **params):
    lev = params['plot']['limit']
    level = params['plot']['barlim']
    label = params['plot']['label']
    ls = params['plot']['labelsize']
    fs = params['plot']['fontsize']

    ### Plot variable data for trends
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
    plt.rc('savefig', facecolor='white')
    plt.rc('axes', edgecolor='darkgrey')
    # plt.rc('xtick', color='darkgrey')
    # plt.rc('ytick', color='darkgrey')
    plt.rc('xtick', color='black')
    plt.rc('ytick', color='black')
    plt.rc('axes', labelcolor='darkgrey')
    plt.rc('axes', facecolor='white')

    yearperiod = data.periods.values
    datasets = data.models.values
    wd_ratio = np.concatenate((np.repeat(np.array([4]), data.shape[1]),np.array([1])))


    fig1, axes = plt.subplots(nrows=data.shape[0], ncols=data.shape[1]+1, figsize=params['plot']['figsize'],
                              subplot_kw={'projection': ccrs.Robinson(central_longitude=0)},
                              gridspec_kw={"width_ratios": wd_ratio,
                                           "hspace": 0.1})


    m = ccrs.PlateCarree()
    c = 0
    for mod in range(data.shape[0]):

        uai = data.loc[data.models.values[mod]]

        for yp in range(axes.shape[1]):
            ax = axes[mod, yp]

            if uai.min().values <0:
                cmp = "seismic"
                ls = 10
            else:
                cmp = "Reds"
                ls = 5
            if params['plot']['add_raw'] and mod == data.shape[0]-1:
                    cmp = params['plot']['cmap']

            if yp == data.shape[1]:
                axes[mod, yp].axis('off')

                if mod < (data.shape[0] - 1):
                    # level1 = np.asarray(np.linspace(uai.min().values, uai.max().values, 8))
                    # lev1 = np.around(level1, 2).astype(str)
                    # vp1.set_clim(uai.min().values, uai.max().values)
                    # cbar_ax = fig1.add_axes(params['plot']['axis'])
                    # cbar1 = fig1.colorbar(vp1, cax=cbar_ax, orientation='vertical')
                    cbar1 = fig1.colorbar(vp1, ax=ax, orientation='vertical')
                    cbar1.ax.tick_params(axis='x', size=.04, labelsize=fs)
                    cbar1.outline.set_edgecolor('darkgrey')
                    # cbar1.set_ticks(level1)
                    # cbar1.set_ticklabels(lev1)
                    cbar1.set_label(label, fontsize=fs, color='black', labelpad=1.4)

            else:
                ax.coastlines(resolution='110m', color='dimgrey', linewidth=0.35)
                ax.set_global()
                if params['plot']['types'] == 'continuous':

                    if uai.min().values < 0:
                        xmax = np.max(np.array([np.abs(uai.min().values), uai.max().values]))
                        if data.models.values[mod] == 'LRP':
                            vp = uai[yp, :, :].plot(ax=ax, transform=m, cmap=cmp, vmin=0.2, vmax=xmax,
                                                    add_colorbar=False, add_labels=False)
                        else:
                            vp = uai[yp, :, :].plot(ax=ax, transform=m, cmap=cmp, vmin=-xmax, vmax=xmax,
                                                    add_colorbar=False, add_labels=False)

                    else:
                        vp = uai[yp, :, :].plot(ax=ax, transform=m, cmap=cmp, vmin=uai.min().values,
                                                vmax=uai.max().values,
                                                add_colorbar=False, add_labels=False)

                    if mod < (data.shape[0] - 1) and yp == (axes.shape[1] - 2):
                        vp1 = vp
                else:
                    if len(uai.shape) > 3:
                        uaim = uai.mean(axis=0)
                    else:
                        uaim = uai
                    xmax = np.max(np.array([np.abs(uaim.min().values), uaim.max().values]))
                    if mod <= data.shape[0]-1:
                        vp = uaim[yp, :, :].plot.pcolormesh(ax=ax, transform=m, cmap=cmp, vmin=-xmax, vmax=xmax, levels=ls,
                                                        add_colorbar=False, add_labels=False)
                    else:
                        vp = uaim[yp, :, :].plot(ax=ax, transform=m, cmap=cmp, vmin=-xmax, vmax=xmax,
                                                add_colorbar=False, add_labels=False)

                    if mod < (data.shape[0] - 1) and yp == (axes.shape[1] - 2):
                        vp1 = vp

                    if mod < (data.shape[0] - 1) and yp == (axes.shape[1] - 1):
                            cbar1 = fig1.colorbar(vp1, ax=axes[mod, yp], orientation='vertical')
                            cbar1.ax.tick_params(axis='x', size=.02, labelsize=ls)
                            cbar1.outline.set_edgecolor('darkgrey')
                            cbar1.set_label(label, fontsize=fs, color='black', labelpad=1.4)

            if mod == data.shape[0]-1 and yp == 0:
                level = np.asarray(np.linspace(uai.min().values, uai.max().values, 8))
                lev = np.around(level, 2).astype(str)


            if (c % axes.shape[1]) == 0:
                ax.annotate(r'\textbf{%s}' % datasets[mod], xy=(0, 0), xytext=(-0.1, 0.5),
                        textcoords='axes fraction', color='black', fontsize=fs,
                        rotation=90, ha='center', va='center')

            if c < len(yearperiod):
                ax.annotate(r'\textbf{%s}' % yearperiod[yp], xy=(0, 0), xytext=(0.5, 1.22),
                            textcoords='axes fraction', color='darkgrey', fontsize=fs,
                            rotation=0, ha='center', va='center')



            c += 1

    ###########################################################################
    cbar = fig1.colorbar(vp, ax=ax,  orientation='vertical')
    if params['plot']['add_raw']:
        labs = r'\textbf{stdized. T}'
    else:
        labs = label
    cbar.set_label(labs, fontsize=fs, color='black', labelpad=1.4)
    cbar.ax.tick_params(axis='x', size=.04, labelsize=fs)
    cbar.outline.set_edgecolor('darkgrey')
    if params['plot']['add_raw']:

        cbar.set_ticks(level)
        cbar.set_ticklabels(lev)

    fig1.subplots_adjust(top=0.85, wspace=0.01, hspace=0, bottom=0.14)
    fig1.savefig(params['plot']['dir'] + params['plot']['figname'], dpi=600, bbox_inches='tight')

    plt.close(fig1)

    return

def timeperiod_calc(
        data: xr.DataArray,
        ** params):
    startY = params['start_year']
    endY = params['end_year']
    step = params['nyears']
    years = np.arange(startY, endY + 1, 1)

    period = []

    comp = np.empty((data.shape[0],data.shape[1],len(years) // step, data.shape[3], data.shape[4]))
    for mod in range(data.values.shape[0]):
        dat = data.values[mod,:,:,:,:]

        for count, i in enumerate(range(0, len(years), step)):
            if mod == 0:
                year_range = str(startY + (count * step)) + '-' + str(startY + ((count + 1) * step))
                period.append(year_range)
            if count < len(years)// step:
                comp[mod, :,count, :, :] = np.nanmean(dat[:,i:i + 40, :,:], axis=1)


    data_period = xr.DataArray(data=comp, dims=["model","ensembles","time", "lat", "lon"],
                      coords=dict(model= np.arange(data.values.shape[0]),ensembles= np.arange(data.values.shape[1]),lon=("lon", data["lon"].values), lat=("lat", data["lat"].values), time=period),
                      attrs=dict(description="T2M", units="K", ))

    return data_period


def plot_singleDataMaps(
        data: xr.DataArray,
        **params):

    ### Plot variable data for trends
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
    plt.rc('savefig', facecolor='white')
    plt.rc('axes', edgecolor='darkgrey')
    # plt.rc('xtick', color='darkgrey')
    # plt.rc('ytick', color='darkgrey')
    plt.rc('xtick', color='black')
    plt.rc('ytick', color='black')
    plt.rc('axes', labelcolor='darkgrey')
    plt.rc('axes', facecolor='white')
    fig1, axes = plt.subplots(nrows=params['plot']['rows'], ncols=params['plot']['cols'], figsize=(24, 10),
                              subplot_kw={'projection': ccrs.Mollweide(central_longitude=0)},
                              )

    m = ccrs.PlateCarree()
    c = 0
    yearperiod = data.time.values
    datasets = data.models.values
    uai = data.loc[data.models.values[0]]
    level = np.asarray(np.arange(uai.min().values, uai.max().values, 0.5))
    lev = np.around(level, 1).astype(str)
    rws = np.arange(0,params['plot']['rows'],1)
    rows = np.repeat(rws[:,np.newaxis],params['plot']['cols'], 1).flatten()
    cls = np.arange(0,params['plot']['cols'],1)
    cols = np.repeat(cls[:,np.newaxis], params['plot']['rows'], 1).T.flatten()

    for yp in range(data.shape[1]):
        ax = axes[rows[c],cols[c]]

        ax.coastlines(resolution='110m', color='dimgrey', linewidth=0.35)
        ax.set_global()

        vp = uai[yp, :, :].plot(ax=ax, transform=m, cmap=params['plot']['cmap'],
                                add_colorbar=False, add_labels=False)

        ax.annotate(r'\textbf{%s}' % yearperiod[yp], xy=(0, 0), xytext=(0.5, 1.22),
                    textcoords='axes fraction', color='darkgrey', fontsize=10,
                    rotation=0, ha='center', va='center')

        c += 1

    ###########################################################################

    cbar_ax = fig1.add_axes([0.32, 0.05, 0.4, 0.03])
    cbar = fig1.colorbar(vp, cax=cbar_ax, orientation='horizontal')

    cbar.ax.tick_params(axis='x', size=.02, labelsize=10)
    cbar.outline.set_edgecolor('darkgrey')
    # cbar.set_clim(level.min(),level.max())
    cbar.set_ticks(level)
    cbar.set_ticklabels(lev)
    cbar.set_label("temp. diff.", fontsize=12, color='black', labelpad=1.4)

    # fig1.subplots_adjust(top=0.85, wspace=0.01, hspace=0, bottom=0.14)
    fig1.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0, hspace=0)

    fig1.savefig(params['plot']['dir'] + params['plot']['figname'], dpi=600)#, bbox_inches='tight')

    plt.close(fig1)


    return