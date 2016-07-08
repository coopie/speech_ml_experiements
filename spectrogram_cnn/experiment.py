import src_context

import learning
from ttv_to_spectrograms import ttv_to_spectrograms
from util import ttv_yaml_to_dict, EMOTIONS
from waveform_tools import pad_or_slice

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling1D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import numpy as np
import math
import random
import os

THIS_DIR = os.path.dirname(os.path.realpath(__file__)) + '/'

def main():
    ttv_info = ttv_yaml_to_dict(THIS_DIR + '../ttvs/ttv_rb.yaml')
    print("GETTING SPECTORGRAM DATA...")
    spectrogram_data = ttv_to_spectrograms(
        ttv_info,
        normalise_waveform=normalise,
        normalise_spectrogram=slice_spectrogram,
        cache=THIS_DIR
    )

    test, train, val = ttv_data = learning.split_ttv(spectrogram_data)

    # test['x']  = np.reshape(test['x'],  (test['x'].shape[0] ,) + (1,) + test['x'].shape[1:] )
    # train['x'] = np.reshape(train['x'], (train['x'].shape[0],) + (1,) + train['x'].shape[1:]  )
    # val['x']   = np.reshape(val['x'],   (val['x'].shape[0]  ,) + (1,) + val['x'].shape[1:]  )


    learning.train(
        make_model,
        ttv_data,
        THIS_DIR + 'model',
        path_to_results=THIS_DIR,
        generate_callbacks=generate_callbacks,
        number_of_epochs=200
    )

def generate_callbacks():
    return [
        EarlyStopping(monitor='val_acc', patience=20)
    ]

def make_model(**kwargs):
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    compile_args = {
        'loss': "categorical_crossentropy",
        'optimizer': sgd,
        'metrics': ['accuracy']
    }

    model = Sequential()
    # if kwargs['verbosity'] >= 1:
        # print('top layer: ', top_layer, 'second layer: ', second_layer)

    input_shape = kwargs['example_input'].shape
    print('input shape: ', input_shape)
    window_thickness = input_shape[2] // 10

    layer = Convolution2D(8, int(input_shape[1] * 0.8), window_thickness, input_shape=input_shape)
    # layer = Convolution2D(8, input_shape[1], 1, input_shape=input_shape)


    # model.add(Convolution2D(1, input_shape[0] -1 , window_thickness, input_shape=input_shape))
    model.add(layer)
    model.add(Activation('relu'))


    model.add(Convolution2D(1, 3, 20))
    model.add(Activation('relu'))
    # # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    #
    # model.add(Convolution2D(1, 10, 10))
    # model.add(Activation('relu'))
    # # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    # add some 1d convolutions?

    model.add(Flatten())
    model.add(Dense(64))


    model.add(Dense(len(EMOTIONS)))
    model.add(Activation('softmax'))

    print('input shape of top layer:', layer.input_shape)
    print('output shape of top layer:', layer.output_shape)

    model.compile(
        **compile_args
    )
    return model, compile_args


def normalise(datum, frequency):
    return pad_or_slice(datum, 3*frequency)


def slice_spectrogram(spec, frequencies,**unused):
    smaller_than_2khz = sum(frequencies < 2000)
    return np.log(spec[:smaller_than_2khz] + 1) / np.log(256) # log base 256 (largest number possible is 2)


if __name__ == '__main__':
    main()
