#!/usr/bin/env python3

from __future__ import print_function
import sys
import os
import tensorflow.keras as keras
import numpy as np
# fix random seed for reproducibility
seed = 42
np.random.seed(seed)
from optparse import OptionParser
import h5py
from tensorflow.keras.optimizers import Adam, Nadam
import tensorflow as tf
from utils.callbacks import all_callbacks
import pandas as pd
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import shutil
import yaml

from qkeras.autoqkeras import *
from qkeras import *
from qkeras.utils import model_quantize
from qkeras.qtools import run_qtools
from qkeras.qtools import settings as qtools_settings
# To turn off GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''


def runAutoQKeras(STRATEGY,train_data,val_data,keras_model):
  
    print("For model with layers:", [layer.name for layer in keras_model.layers])
    goal = {
        "type": "energy",
        "params": {
            "delta_p": 8.0,
            "delta_n": 8.0,
            "rate": 4.0,
            "stress": 0.8,
            "process": "horowitz",
            "parameters_on_memory": ["sram", "sram"],
            "activations_on_memory": ["sram", "sram"],
            "rd_wr_on_io": [False, False],
            "min_sram_size": [0, 0],
            "source_quantizers": ["fp16"],
            "reference_internal": "fp16",
            "reference_accumulator": "fp16"
        }
    }

    limit = {
        "Conv2D":[4,8,6],
        "DepthwiseConv2D":[4,8,6],
        "Dense":[4,8,6],
        "Activation":[16],
        "BatchNormalization":[]
    }

    run_config = {
        "goal": goal,
        "transfer_weights": False,
        "mode": "random",
        "seed": 42,
        "limit": limit,
        "tune_filters": "layer",
        "tune_filters_exceptions": "^output$",
        "learning_rate_optimizer": False,
        "output_dir": "autoqk/",
        "layer_indexes": [3,5,6,9,10,13,15,18],
        "max_trials": 500,
        "distribution_strategy": STRATEGY,
    }
    print("quantizing layers:", [keras_model.layers[i].name for i in run_config["layer_indexes"]])
    
    run_config["quantization_config"] = {
        "kernel": {
                # "binary": 1,
                # "stochastic_binary": 1,
                # "ternary": 2,
                # "stochastic_ternary": 2,
                # "quantized_bits(2,1,1,alpha=1.0)": 2,
                "quantized_bits(4,0,1,alpha=1.0)": 4,
                # "quantized_bits(8,0,1,alpha=1.0)": 8,
                # "quantized_po2(4,1)": 4
        },
        "bias": {
                "quantized_bits(4,0,1)": 4,
                # "quantized_bits(8,3,1)": 8,
                # "quantized_po2(4,8)": 4
        },
        "activation": {
                # "binary": 1,
                # "ternary": 2,
  #               "quantized_relu_po2(4,4)": 4,
  #               "quantized_relu(3,1)": 3,
                # "quantized_tanh(3,1)": 3,
                # "quantized_relu(4,2)": 4,
                # "quantized_tanh(4,2)": 4,
                # "quantized_relu(4,2,negative_slope=0.25)": 4,
                # "quantized_relu(8,2)": 8,
                # "quantized_relu(8,2,negative_slope=0.125)": 8,
                "quantized_relu(8,4)": 8,
                "quantized_relu(16,8)": 16,
                # "quantized_relu(32,6)": 32
        },
        "linear": {
                # "binary": 1,
                # "ternary": 2,
                "quantized_bits(4,1)": 4,
                # "quantized_bits(8,2)": 8,
                # "quantized_bits(16,10)": 16
        }
    }

    metrics = [tf.keras.metrics.AUC(), 'accuracy']

    custom_objects = {}
    if "blocks" in run_config:
      autoqk = AutoQKerasScheduler(
          keras_model, metrics, custom_objects, debug=0, **run_config)
    else:
      # in debug mode we do not run AutoQKeras, just the sequential scheduler.
      autoqk = AutoQKeras(keras_model, metrics, custom_objects, **run_config)

    
    # autoqk.fit(
    #     train_data, batch_size=1024, epochs=60,
    #     validation_split = 0.25, shuffle = True)
    autoqk.fit(
        train_data, epochs=60,
        validation_data = val_data, shuffle = True)

    qmodel = autoqk.get_best_model()

    metrics = [tf.keras.metrics.AUC(), 'accuracy']
    adam = Adam(lr=startlearningrate)
    qmodel.compile(optimizer=adam, loss=[yamlConfig['KerasLoss']], metrics=metrics)
    save_quantization_dict("cern.dict", qmodel)

    callbacks=all_callbacks(stop_patience=1000,
                            lr_factor=0.5,
                            lr_patience=10,
                            lr_epsilon=0.000001,
                            lr_cooldown=2,
                            lr_minimum=0.0000001,
                            outputDir=options.outputDir)

    qmodel.fit(
        X_train_val, y_train_val, batch_size=1024, epochs=1000,
        validation_split = 0.25, shuffle = True, callbacks=callbacks.callbacks)

    qmodel.save("auto.h5")

