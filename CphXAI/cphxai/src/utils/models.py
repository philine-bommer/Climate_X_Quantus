from typing import Any, Tuple

import numpy as np
import keras.backend as K
import keras.layers
from keras.layers import Dense, Activation, Input, Conv2D, MaxPooling2D
from keras import regularizers
from keras import metrics
from keras import optimizers
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet_v2 import MobileNetV2




def defineNN(
        input_shape: float,
        output_shape: float,
        **params) -> Sequential:

    hidden = params.get('hidden_layer',20)
    random_network_seed = params.get('random_network_seed', None)
    actFun = params.get('actFun', 'relu')
    ridgePenalty = params.get('penalty', 0.01)

    model = Sequential()
    ### Initialize first layer
    if hidden[0] == 0:
        ### Model is linear
        model.add(Dense(1, input_shape=(input_shape,),
                        activation='linear', use_bias=True,
                        kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=ridgePenalty),
                        bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                        kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))
        print('\nTHIS IS A LINEAR NN!\n')
    else:
        ### Model is a single node with activation function
        model.add(Dense(hidden[0], input_shape=(input_shape,),
                        activation=actFun, use_bias=True,
                        kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=ridgePenalty),
                        bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                        kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))

        ### Initialize other layers
        for layer in hidden[1:]:
            model.add(Dense(layer, activation=actFun,
                            use_bias=True,
                            kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=0.00),
                            bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                            kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))

        print('\nTHIS IS A ANN!\n')

    #### Initialize output layer
    model.add(Dense(output_shape, activation=None, use_bias=True,
                    kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=0.0),
                    bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                    kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))

    ### Add softmax layer at the end
    model.add(Activation('softmax'))

    return model

def defineCNN(
        input_shape: tuple,
        output_shape: float,
        **params) -> Sequential:

    hidden = params.get('filters', 1024)
    random_network_seed = params.get('random_network_seed', None)
    not_trainable = params['train']['not_train']


    ### mark loaded layers as not trainable
    # i = 0
    if params['train']['nets'] == '2layer':

        inputs = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
        x = Conv2D(6, (6, 6), padding='same')(inputs)
        x = MaxPooling2D((2, 2), strides=2)(x)
        flat1 = keras.layers.Flatten()(x)
        class1 = Dense(hidden, activation='relu', name='dense_0')(flat1)
        output = Dense(output_shape, activation='softmax', use_bias=True,
                        kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=0.0),
                        bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                        kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed), name='dense_out')(class1)
        model = keras.Model(inputs, output)
        model.summary()

    else:
        if params['train']['nets'] == 'vgg':
            vgg = VGG16(include_top=False, input_shape=(input_shape[0], input_shape[1], 3))
            if params['train']['dropout']:
                for layer in vgg.layers:
                    layer.rate = params['train']['dropout']
                base_model = vgg
            else:
                base_model = vgg
            hidden = params.get('hiddens', 1024)
        if params['train']['nets'] == 'mnV2':
            mnv2 = MobileNetV2(include_top=False, input_shape=(input_shape[0], input_shape[1], 3))
            if params['train']['dropout']:
                for layer in mnv2.layers:
                    layer.rate = params['train']['dropout']
                base_model = mnv2
            else:
                base_model = mnv2
            hidden = 1000

        base_model.summary()
        base_model.trainable = False
        inputs = Input(shape=(input_shape[0],input_shape[1],input_shape[2]))
        x = Conv2D(3, (3, 3), padding='same')(inputs)
        x = base_model(x)
        base_model.summary()
        flat1 = keras.layers.Flatten()(x)
        # flat1 = tf.keras.layers.flatten()(x)
        class1 = Dense(hidden, activation='relu')(flat1)
        output = Dense(output_shape, activation='softmax', use_bias=True,
                        kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=0.0),
                        bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                        kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed))(class1)
        model = keras.Model(inputs, output)



    return model

