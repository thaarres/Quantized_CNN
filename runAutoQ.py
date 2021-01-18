#!/usr/bin/env python3
import os, sys
import logging
import math
import h5py
import kerastuner as kt
import tensorflow as tf
import argparse
import seaborn as sns

import tensorflow_datasets as tfds
AUTO = tf.data.experimental.AUTOTUNE
  
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from qkeras import QConv2D, QDense, Clip, QActivation
from qkeras.autoqkeras import *
from qkeras import *
from qkeras.utils import model_quantize
from qkeras.qtools import run_qtools
from qkeras.qtools import settings as qtools_settings
from tensorflow.keras.callbacks import Callback, EarlyStopping,History,ModelCheckpoint,TensorBoard,ReduceLROnPlateau,TerminateOnNaN,LearningRateScheduler

import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
import tempfile

np.random.seed(20)
DATASET = "svhn_cropped" 
batchsize = 512  

doLROpt = False 
train = True
doChecks = True
odir = 'AUTOQ_v5'
if not os.path.exists(odir):
    os.system('mkdir {}'.format(odir))

custom_objects = {'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation}

def plot_convolutional_filters(img):
    
    img = np.expand_dims(img, axis=0)
    activations = activation_model.predict(img)
    images_per_row = 9
    
    for layer_name, layer_activation in zip(layer_names, activations): 
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols): 
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                 :, :,
                                                 col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.clf()
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='plasma')
        plt.savefig('AUTOQ/{}_filters.pdf'.format(layer_name))

def getConfusion(model,X_train,y_train):
  
  # Plot the confusion matrix
  
  matrix = confusion_matrix(y_train, y_pred, labels=[0,1,2,3,4,5,6,7,8,9])
  plt.clf()
  fig, ax = plt.subplots(figsize=(14, 12))
  sns.heatmap(matrix, annot=True, cmap='Greens', fmt='d', ax=ax)
  plt.title('Confusion Matrix for training dataset')
  plt.xlabel('Predicted label')
  plt.ylabel('True label')
  plt.show()
  plt.savefig('AUTOQ/confusion.pdf')

def get_data(dataset_name, fast=False):
  """Returns dataset from tfds."""
  ds_train = tfds.load(name=dataset_name, split="train", batch_size=-1, data_dir='/afs/cern.ch/user/t/thaarres/tensorflow_datasets/')
  ds_test = tfds.load(name =dataset_name, split="test", batch_size=-1, data_dir='/afs/cern.ch/user/t/thaarres/tensorflow_datasets/')

  dataset = tfds.as_numpy(ds_train)
  x_train, y_train = dataset["image"].astype(np.float32), dataset["label"]
  plt.imshow(x_train[0])
  plt.show()
  from time import sleep; sleep(100)
  plt.savefig('{}/x.pdf'.format(odir))
  dataset = tfds.as_numpy(ds_test)
  x_test, y_test = dataset["image"].astype(np.float32), dataset["label"]
  # print('Train', x_train.min(), x_train.max(), x_train.mean(), x_train.std())
  # print('Test' , x_test.min(), x_test.max(), x_test.mean(), x_test.std())
  if len(x_train.shape) == 3:
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

  x_train /= 255.0
  x_test /= 255.0

  #Get mean and std over all images and subtract/divide
  x_mean = np.mean(x_train, axis=0)
  x_std  = np.std(x_train, axis = 0)
  x_train = (x_train-x_mean)/x_std
  x_test  = (x_test-x_mean)/x_std

  nb_classes = np.max(y_train) + 1
  y_train = to_categorical(y_train, nb_classes)
  y_test = to_categorical(y_test, nb_classes)
  #
  print(x_train.shape[0], "train samples")
  print(x_test.shape[0], "test samples")
  return (x_train, y_train), (x_test, y_test)
  
def getEnergy(model):
  reference_internal = "fp32"
  reference_accumulator = "fp32"

  q = run_qtools.QTools(
      model,
      process="horowitz",
      source_quantizers=[quantized_bits(16, 6, 1)],
      is_inference=False,
      weights_path=None,
      keras_quantizer=reference_internal,
      keras_accumulator=reference_accumulator,
      for_reference=True)
  
  energy_dict = q.pe(
    weights_on_memory="fixed",
    activations_on_memory="fixed",
    min_sram_size=8*16*1024*1024,
    rd_wr_on_io=False)

  energy_profile = q.extract_energy_profile(qtools_settings.cfg.include_energy, energy_dict)
  total_energy = q.extract_energy_sum(qtools_settings.cfg.include_energy, energy_dict)
  
  pprint.pprint(energy_profile)
  print()
  print("Total energy: {:.2f} uJ".format(total_energy / 1000000.0))

if __name__ == "__main__":

  model = tf.keras.models.load_model("models/full_0/model_best.h5",custom_objects=custom_objects)
  model.summary()
  # config = blmodel.get_config()
  # model = tf.keras.Model.from_config(config)

  (x_train, y_train), (x_test, y_test) = get_data(DATASET)
  # datagen = ImageDataGenerator(rescale=1.0/255.0,featurewise_center=True, featurewise_std_normalization=True)
  # datagen = ImageDataGenerator( rotation_range=8,
  #                               zoom_range=[0.95, 1.05],
  #                               height_shift_range=0.10,
  #                               shear_range=0.15)
  # datagen.fit(x_train)
  # print(datagen.mean)
  X_train, X_val, y_train, y_val = train_test_split(x_train, y_train,
                                                    test_size=0.15, random_state=22)
                                                  
  spe = X_train.shape[0]// batchsize
  optimizer = Adam(lr=3E-3, amsgrad=True)
  model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"])

  all_callbacks = [
      tf.keras.callbacks.EarlyStopping(patience=8),
      tf.keras.callbacks.ModelCheckpoint(filepath='{}/KERAS_best.h5'.format(odir),monitor="val_loss",verbose=0,save_best_only=True), 
      tf.keras.callbacks.ModelCheckpoint(filepath='{}/KERAS_best_weights.h5'.format(odir),monitor="val_loss",verbose=0,save_weights_only=True),
      tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=2, min_lr=1E-6)   
  ]

  if doLROpt:
    all_callbacks.append(lr_schedule)
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
                  lambda epoch: 1e-4 * 10**(epoch / 10))

  if train:
    history = model.fit(X_train, y_train, epochs=10, batch_size=batchsize, steps_per_epoch=spe, validation_data=(X_val, y_val),callbacks=all_callbacks)
    # history = model.fit(datagen.flow(X_train, y_train, batch_size=batchsize), steps_per_epoch=spe,
    #                             epochs=50, validation_data=(X_val, y_val),
    #                            callbacks=all_callbacks)
  else:
    model = tf.keras.models.load_model('{}/KERAS_best.h5'.format(odir),custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation})
                               
  model.load_weights('{}/KERAS_best_weights.h5'.format(odir))                                                    
  if doLROpt:
    plt.semilogx(history.history['lr'], history.history['loss'])
    plt.axis([1e-4, 1e-1, 0, 1])
    plt.xlabel('Learning Rate')
    plt.ylabel('Training Loss')
    plt.show()
    plt.savefig('AUTOQ/lr_scan.pdf')

  test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

  print('Test accuracy is: {:0.4f} \nTest loss is: {:0.4f}'.format(test_acc, test_loss))

  if doChecks:
    y_pred = model.predict(X_train)

    y_pred  = np.argmax(y_pred, axis=1)
    y_train = np.argmax(y_train, axis=1)
  
    getConfusion(model,X_train,y_train)

    layers = [model.get_layer('conv_0'), 
              model.get_layer('conv_1'), 
              model.get_layer('conv_2')
              ]

    layer_outputs = [layer.output for layer in layers]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

    layer_names = []
    for layer in layers:
        layer_names.append(layer.name)

    img = X_train[5]
    print(y_train[5])
    plt.clf()
    plt.imshow(img)
    plt.show()
    plt.savefig('{}/predict.pdf'.format(odir))
    plot_convolutional_filters(img)
    
  # quantization_config = {
  #       "kernel": {
  #               "binary": 1,
  #               "ternary": 2,
  #               "quantized_bits(2,1,1,alpha=1.0)": 2,
  #               "quantized_bits(4,0,1,alpha=1.0)": 4,
  #               "quantized_bits(8,0,1,alpha=1.0)": 8,
  #               # "quantized_po2(4,1)": 4
  #       },
  #       "bias": {
  #               "binary": 1,
  #               "ternary": 2,
  #               "quantized_bits(4,0,1)": 4,
  #       },
  #       "activation": {
  #               "binary": 1,
  #               "ternary": 2,
  #               # "quantized_relu_po2(4,4)": 4,
  #               "quantized_relu(3,1)": 3,
  #               "quantized_relu(4,2)": 4,
  #               "quantized_relu(8,2)": 8,
  #               "quantized_relu(8,4)": 8,
  #               "quantized_relu(16,6)": 16
  #       },
  #       "linear": {
  #               "binary": 1,
  #               "ternary": 2,
  #               "quantized_bits(4,1)": 4,
  #               "quantized_bits(8,2)": 8,
  #               "quantized_bits(16,6)": 16
  #       }
  # }
  #
  # limit = {
  # "Dense": [8, 8, 8],
  # "Conv2D": [8, 8, 8],
  # "Activation": [8]
  # # "BatchNormalization": []
  # }
  #
  # goal = {
  #   "type": "energy",
  #   "params": {
  #       "delta_p": 4.0,
  #       "delta_n": 4.0,
  #       "rate": 2.0,
  #       "stress": 1.0,
  #       "process": "horowitz",
  #       "parameters_on_memory": ["sram", "sram"],
  #       "activations_on_memory": ["sram", "sram"],
  #       "rd_wr_on_io": [False, False],
  #       "min_sram_size": [0, 0],
  #       "source_quantizers": ["fp16"],
  #       "reference_internal": "fp16",
  #       "reference_accumulator": "fp16"
  #       }
  # }
  #
  # run_config = {
  # "output_dir": "{}/".format(odir),
  # "goal": goal,
  # "quantization_config": quantization_config,
  # "learning_rate_optimizer": False,
  # "transfer_weights": False,
  # "mode": "bayesian",
  # "seed": 42,
  # "limit": limit,
  # "tune_filters": "layer",
  # "tune_filters_exceptions": "output*",
  # "layer_indexes": range(1, len(model.layers) - 1),
  # "max_trials": 500,
  # "blocks": [
  #     "c.*_0$",
  #     "c.*_1$",
  #     "c.*_2$",
  #     "d.*_0$",
  #     "d.*_1$",
  #     "output_dense",
  #   ],
  #   "schedule_block": "cost"
  #
  # }
  #
  #
  # all_callbacks = [
  #     tf.keras.callbacks.EarlyStopping(patience=6),
  #     tf.keras.callbacks.ModelCheckpoint(filepath='{}/AUTOQKERAS_best.h5'.format(odir),monitor="val_loss",verbose=0,save_best_only=True),
  #     tf.keras.callbacks.ModelCheckpoint(filepath='{}/AUTOQKERAS_best_weights.h5'.format(odir),monitor="val_loss",verbose=0,save_weights_only=True),
  #     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=2, min_lr=1E-6)
  #   ]
  #
  # print("quantizing layers:", [model.layers[i].name for i in run_config["layer_indexes"]])
  # model.summary()
  # autoqk = AutoQKerasScheduler(model, metrics=["acc"], custom_objects=custom_objects, debug=False, **run_config)
  # autoqk.fit(X_train, y_train, validation_data=(x_test, y_test), batch_size=batchsize, epochs=20,callbacks=all_callbacks)
  # qmodel = autoqk.get_best_model()
  # qmodel.save('{}/best_pretrain.h5'.format(odir))
  # all_callbacks = [
  #     tf.keras.callbacks.EarlyStopping(patience=8),
  #     tf.keras.callbacks.ModelCheckpoint(filepath='{}/FINAL.h5'.format(odir),monitor="val_loss",verbose=0,save_best_only=True),
  #     tf.keras.callbacks.ModelCheckpoint(filepath='{}/FINAL_weights.h5'.format(odir),monitor="val_loss",verbose=0,save_weights_only=True),
  #     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=2, min_lr=1E-6)
  # ]
  # history = qmodel.fit(X_train, y_train, epochs=100, batch_size=batchsize, steps_per_epoch=spe, validation_data=(X_val, y_val),callbacks=all_callbacks)
