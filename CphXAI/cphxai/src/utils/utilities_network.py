from typing import Optional, Any, Union, Dict
import time
import matplotlib.pyplot as plt
import numpy as np
# import keras.backend as K
# import tensorflow as tf
import pandas as pd
import random
import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
import tensorflow.compat.v1.keras as keras
from keras.layers import Dense, Activation
from keras import regularizers
from keras import metrics
from keras import optimizers
from keras.models import Sequential

from ..utils.models import *
from ..utils.utilities_calc import *
from ..utils.utilities_data import *


################################################################################
###############################################################################
### Neural Network Creation & Training

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def trainNN(
        model: keras.models.Sequential,
        Xtrain: np.ndarray,
        Ytrain: np.ndarray,
        **params):

    niter = params.get('niter', 500)
    verbose = params['train']['verbose']
    actFun = params.get('actFun', 'relu')

    lr_here = params.get('lr', 0.01)
    batch_size = params.get('batch_size', 32)

    # lr_here = .01
    model.compile(optimizer=optimizers.SGD(lr=lr_here,
                                           momentum=0.9, nesterov=True),  # Adadelta .Adam()
                  loss='binary_crossentropy',
                  metrics=[metrics.categorical_accuracy], )



    print('----ANN Training: learning rate = ' + str(
        lr_here) + '; activation = ' + actFun + '; batch = ' + str(batch_size) + '----')
    time_callback = TimeHistory()
    history = model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=niter,
                        shuffle=True, verbose=verbose,
                        callbacks=[time_callback],
                        validation_split=0.2)
    print('******** done training ***********')

    return model, history

def trainGeneral(
        model: keras.models.Sequential,
        Xtrain: np.ndarray,
        Ytrain: np.ndarray,
        Xtest: np.ndarray,
        Ytest: np.ndarray,
        **params):

    niter = params.get('niter', 500)
    epochs = params['train']['epochs']
    verbose = params['train']['verbose']
    actFun = params.get('actFun', 'relu')

    lr_here = params.get('lr', 0.01)
    batch_size = params.get('batch_size', 32)

    # lr_here = .01

    model.compile(optimizer=optimizers.SGD(lr=lr_here,
                                           momentum=0.9, nesterov=True),  # Adadelta .Adam()
                  loss=params['train']['loss'],
                  metrics=[metrics.categorical_accuracy], )

    ### Declare the relevant model parameters

    print('----ANN Training: learning rate = ' + str(
        lr_here) + '; activation = ' + actFun + '; batch = ' + str(batch_size) + '----')
    time_callback = TimeHistory()
    history = model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=niter,
                        shuffle=True, verbose=verbose,
                        callbacks=[time_callback],
                        validation_split=0.2)
    # pdb.set_trace()
    print('******** done training ***********')

    return model, history



def test_train_loopClass(
        Xtrain: np.ndarray,
        Ytrain: np.ndarray,
        Xtest: np.ndarray,
        Ytest: np.ndarray,
        params: Dict[str, Union[Optional[int], Optional[float], Optional[bool], Any]]):
    """or loops to iterate through training iterations, ridge penalty,
    and hidden layer list
    """
    results = {}
    # global nnet, random_network_seed
    iterations = params.get('iterations', 500)
    ridge_penalty = params.get('penalty', 0.01)
    hiddens = params.get('hiddens',[20,20])
    plot_in_train = params.get('plot', False)
    startYear = params.get('starYear', 1921)
    classChunk = params.get('classChunk', 10)
    rm_annual_mean = params.get('rm_annual_mean', False)
    rm_merid_mean = params.get('rm_merid_mean', False)
    land_only = params.get('land_only', False)
    ocean_only = params.get('ocean_only', False)
    random_segment_seed = params.get('random_segment_seed', None)
    yearsall = params.get('yearsall', 'timelens')
    random_network_seed = params.get('random_network_seed', None)
    experiment_result = params.get('experiment_result', [])
    batchsize = params.get('batch_size', 32)

    for niter in iterations:
        for penalty in ridge_penalty:
            params['penalty'] = penalty
            for hidden in hiddens:

                ### Check / use random seed

                np.random.seed(random_network_seed)
                random.seed(random_network_seed)
                tf.set_random_seed(0)

                params['random_network_seed'] = random_network_seed

                ### Standardize the data
                Xtrain, Xtest, stdVals = standardize_data(Xtrain, Xtest)
                Xmean, Xstd = stdVals

                ### Define the model
                params['niter'] = niter
                if params['net'] == 'MLP':
                    params['hidden_layer'] = hidden
                    model = defineNN(
                                 np.shape(Xtrain)[1],
                                 np.shape(Ytrain)[1],
                                 **params)
                    model, history = trainNN(model, Xtrain, Ytrain, **params)
                elif params['net'] == 'CNN':
                    params['filters'] = 32

                    model = defineCNN(
                                 input_shape=(Xtrain.shape[1],Xtrain.shape[2],1),
                                 output_shape=np.shape(Ytrain)[1],
                                 **params)

                    model, history = trainGeneral(model, Xtrain, Ytrain, Xtest, Ytest, **params)

                result = model.evaluate(Xtest, Ytest, batch_size = batchsize)
                print("test loss, test acc:", result)

                ### After training, use the network with training data to
                ### check that we don't have any errors and output RMSE
                rmse_train = rmse(convert_fuzzyDecade_toYear(Ytrain, startYear,
                                                                 classChunk,yearsall),
                                      convert_fuzzyDecade_toYear(model.predict(Xtrain),
                                                                 startYear,
                                                                 classChunk,yearsall))
                if type(Ytest) != bool:
                    rmse_test = 0.
                    rmse_test = rmse(convert_fuzzyDecade_toYear(Ytest,
                                                                    startYear, classChunk,yearsall),
                                         convert_fuzzyDecade_toYear(model.predict(Xtest),
                                                                    startYear,
                                                                    classChunk,yearsall))
                else:
                    rmse_test = False

                this_result = {'iters': niter,
                               'hiddens': hidden,
                               'RMSE Train': rmse_train,
                               'RMSE Test': rmse_test,
                               'ridge penalty': penalty,
                               'zero mean': rm_annual_mean,
                               'zero merid mean': rm_merid_mean,
                               'land only?': land_only,
                               'ocean only?': ocean_only,
                               'Segment Seed': random_segment_seed,
                               'Network Seed': random_network_seed,
                               'Model History': history.history,
                               'Training Results': result}
                results.update(this_result)


                experiment_result = experiment_result.append(results,
                                                             ignore_index=True)

                # if True to plot each iter's graphs.
                if plot_in_train == True:
                    plt.figure(figsize=(16, 6))

                    plt.subplot(1, 2, 1)
                    plt.plot(history.history['loss'], label='training')
                    plt.title(history.history['loss'][-1])
                    plt.xlabel('epoch')
                    plt.xlim(2, len(history.history['loss']) - 1)
                    plt.legend()

                    plt.subplot(1, 2, 2)

                    plt.plot(convert_fuzzyDecade_toYear(Ytrain, startYear,
                                                        classChunk,yearsall),
                             convert_fuzzyDecade_toYear(model.predict(Xtrain),
                                                        startYear,
                                                        classChunk,yearsall), 'o',
                             color='gray')
                    plt.plot(convert_fuzzyDecade_toYear(Ytest, startYear,
                                                        classChunk,yearsall),
                             convert_fuzzyDecade_toYear(model.predict(Xtest),
                                                        startYear,
                                                        classChunk,yearsall), 'x',
                             color='red')
                    plt.plot([startYear, yearsall.max()], [startYear, yearsall.max()], '--k')
                    plt.yticks(np.arange(yearsall.min(), yearsall.max(), 10))
                    plt.xticks(np.arange(yearsall.min(), yearsall.max(), 10))

                    plt.grid(True)
                    plt.show()

                # 'unlock' the random seed
                np.random.seed(None)
                random.seed(None)
                tf.set_random_seed(None)

    return experiment_result, model

def generate_random_seed(params):
    '''Genrate list of random seeds for ensemble sampling of the networks, i.e. bayesian approximation'''

    nsample = params['SAMPLEQ']
    random_seed = np.zeros((1,nsample))

    for i in range(nsample):

        random_network_seed = int(np.random.randint(1, 100000))
        random_seed[0,i] = random_network_seed


    return random_seed

