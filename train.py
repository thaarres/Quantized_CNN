#!/usr/bin/env python3

"""Training of 2D quantized CNN (with kFold cross validation)
Author: Thea Aarrestad

Training of floating point precision and quantized deep convolutional neural networks
"""
import os
print("Set TF logging level to minimum (INFO and WARNING messages are not printed)")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #This is buggy after switching to absl, set manually with SET TF_CPP_MIN_LOG_LEVEL=3

import numpy as np
np.random.seed(1337)  # for reproducibility
import sys, os
import tempfile
import json
import pandas as pd
import time
from absl import app
from absl import flags
from absl import logging
FLAGS = flags.FLAGS

print("Importing TensorFlow")
import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping,History,ModelCheckpoint,TensorBoard,ReduceLROnPlateau,TerminateOnNaN,LearningRateScheduler

print("Using TensorFlow version: {}".format(tf.__version__))
print("Using Keras version: {}".format(tf.keras.__version__))
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
print("Forcing image data format to {}".format(K.image_data_format()))
import tensorflow_model_optimization as tfmot

from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks

import tensorboard

import h5py
from qkeras.utils import model_quantize
from qkeras.estimate import print_qstats

import models
from utils import getKfoldDataset, toJSON, preprocess, print_model_sparsity
from qdictionaries import allQDictionaries

flags.DEFINE_string ('outdir'    , None , 'Output directory')
flags.DEFINE_list   ('Filters'   , None , 'Filters   ')
flags.DEFINE_list   ('Kernel'    , None , 'Kernel    ')
flags.DEFINE_list   ('Strides'   , None , 'Strides   ')
flags.DEFINE_list   ('Pooling'   , None , 'Pooling   ')
flags.DEFINE_list   ('Dropout'   , None , 'Pooling   ')
flags.DEFINE_string ('Activation', "relu", 'Activation')
flags.DEFINE_string ('KerasModel', None, 'KerasModel')
flags.DEFINE_integer('folds'    , 10, 'folds', lower_bound=1)
flags.DEFINE_integer('epochs'    , 100, 'Nepochs', lower_bound=1)
flags.DEFINE_integer('batchsize' , 512, 'batchsize', lower_bound=1)
flags.DEFINE_integer('buffersize',1024, 'buffersize', lower_bound=100)
flags.DEFINE_integer('nclasses'  ,10, 'nclasses', lower_bound=1)
flags.DEFINE_boolean('prune', False, 'Prune model (dense only and full model pruning)')
flags.DEFINE_boolean('quantize', False, 'Quantize model')

def setWeights(model,full_model_path):
  full_model = tf.keras.models.load_model(full_model_path)
  for layerA,layerB in zip(model.layers,full_model.layers):
    layerA.set_weights(layerB.get_weights())

def getPrunedModels(fold,input_shape):
  
  pruning_params = {'pruning_schedule': sparsity.ConstantSparsity(0.75, begin_step=2000, frequency=100)}

  model_for_layerwise_pruning = getModel("layerwise_pruning_%i"%fold,"float_cnn_densePrune" , input_shape,pruning_params) 
  
  model_for_1L_pruning   = getModel("1L_pruning_%i"%fold  ,"float_cnn_1L_Prune"   , input_shape,pruning_params) 
  model_for_full_pruning = getModel("full_pruning_%i"%fold,"float_cnn_allPrune" , input_shape,pruning_params) 

  return [model_for_layerwise_pruning,model_for_full_pruning,model_for_1L_pruning]
  
def getQuantizedModel(fold,input_shape,full_model_path="one_hot_v2/full_0/saved_model.h5"):
  qmodels = []
  
  
  try:
    model = tf.keras.models.load_model(full_model_path)
  except:
    model = getModel("full_%i"%fold,FLAGS.KerasModel, input_shape)  
  for name, dict_ in allQDictionaries.items():
    qmodel =  model_quantize(model, dict_, 4, transfer_weights=True)   
    qmodel._name = 'quantized_%s_%i'%(name,fold)
    qmodels.append(qmodel)                  
  return qmodels                 

def getModel(name,modelName,input_shape,options={}):
  
  model = getattr(models, modelName)
  model = model(name, Input(input_shape),
                        FLAGS.nclasses  ,
                        FLAGS.Filters   ,
                        FLAGS.Kernel    ,
                        FLAGS.Strides   ,
                        FLAGS.Pooling   ,
                        FLAGS.Dropout   ,
                        FLAGS.Activation,
                        options)
  return model
  
def getCallbacks(outdir_):
  earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto',min_delta=0.001)
  mcp_save_m    = ModelCheckpoint(outdir_+'/bestModel.h5', save_best_only=True, monitor='val_loss', mode='auto')
  mcp_save_w    = ModelCheckpoint(outdir_+'/bestWeights.h5', save_best_only=True,save_weights_only=True, monitor='val_loss', mode='auto')
  # tensorboard   = tf.keras.callbacks.TensorBoard(log_dir=outdir+'/logs/', update_freq='batch')
  reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_delta=1e-4, mode='auto')
  
  if outdir_.find('quantized')==-1: 
    return [earlyStopping,mcp_save_m,mcp_save_w,reduce_lr_loss]
    
  return [earlyStopping, mcp_save_m,mcp_save_w,reduce_lr_loss]

def fitModels(models,train_data, val_data):
  for model in models:
    callbacks = getCallbacks(FLAGS.outdir+'/%s/'%model.name)
    if FLAGS.quantize == True:
      callbacks = [ ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_delta=1e-4, mode='auto'),
                    EarlyStopping(monitor='val_loss', patience=7, verbose=1, min_delta=1e-4, mode='auto'),
                    ModelCheckpoint(FLAGS.outdir+'/%s/bestModel.h5'%model.name, save_best_only=True, monitor='val_loss', mode='auto'),
                    ModelCheckpoint(FLAGS.outdir+'/%s/bestWeights.h5'%model.name, save_best_only=True,save_weights_only=True, monitor='val_loss', mode='auto')]
      OPTIMIZER   = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)              
    
    if FLAGS.prune == True:
      callbacks = [ pruning_callbacks.UpdatePruningStep(), 
                    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_delta=1e-4, mode='auto'),
                    EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto',min_delta=0.001),
                    #pruning_callbacks.PruningSummaries(log_dir=FLAGS.outdir+'/logs_%s/'%model.name),
                    ModelCheckpoint(FLAGS.outdir+'/%s/bestModel.h5'%model.name, save_best_only=True, monitor='val_loss', mode='auto'),
                    ModelCheckpoint(FLAGS.outdir+'/%s/bestWeights.h5'%model.name, save_best_only=True,save_weights_only=True, monitor='val_loss', mode='auto')]
    start = time.time()
    print("Training model {}".format(model.name))
    history = model.fit(train_data,
                        epochs =  FLAGS.epochs, 
                        validation_data=val_data,
                        callbacks=callbacks,
                        verbose=1)                    
    model.save(FLAGS.outdir+'/%s/saved_model.h5'%(model.name))
    history_dict = history.history
    pd.DataFrame.from_dict(history.history).to_csv(FLAGS.outdir+'/%s/history_dict.csv'%model.name,index=False)
    val_score = model.evaluate(val_data)
    print("Done training model {}".format(model.name))
    print('\n Test loss:', val_score[0])
    print('\n Test accuracy:', val_score[1])
    np.savez(FLAGS.outdir+'/%s/scores'%model.name, val_score)
        
    del model
    end = time.time()
    print('It took {} minutes to train!\n'.format( (end - start)/60.))
  
    
def buildModels(fold, input_shape, train_data, val_data,steps_per_epoch,eval_steps_per_epoch):
  models = []
  if FLAGS.quantize == True:
    models = getQuantizedModel(fold,input_shape)
    
  elif FLAGS.prune == True:
    models = getPrunedModels(fold,input_shape)
    for prunedModel in models:
      setWeights(prunedModel,full_model_path="one_hot_v2/full_0/saved_model.h5")
      
  else:
    models = [getModel("full_%i"%fold,FLAGS.KerasModel, input_shape)]

  for i,model in enumerate(models):
    
    model.summary()
    if FLAGS.quantize == True:
      print_qstats(model)
    
    if not os.path.exists(FLAGS.outdir+'/%s/'%model.name):
      os.system('mkdir '+FLAGS.outdir+'/%s/'%model.name)
    toJSON(model,FLAGS.outdir + '/%s/model.json'%model.name)
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=["accuracy"])
  return models          
    
      
def train(train_data_list, val_data_list, input_shape,train_size):
  
  for i,(val_,train_) in enumerate(zip(val_data_list, train_data_list)): 
    print("Working on fold: {}".format(i))
    train_data = train_.map(preprocess).shuffle(FLAGS.buffersize).batch(FLAGS.batchsize)#.repeat() #see https://www.tensorflow.org/guide/data
    val_data   = val_ .map(preprocess).batch(FLAGS.batchsize)
    # Read a single batch of examples from the training set and display shapes.
    for img_feature, label in train_data:
      break
    print('INPUT img_feature.shape (batch_size, image_height, image_width) =', img_feature.shape)
    print('INPUT label.shape (batch_size, number_of_labels) =', label.shape)
    
    steps_per_epoch      = int(train_size*0.9)  // FLAGS.batchsize #90% train, 10% validation in 10-fold xval
    eval_steps_per_epoch = int(train_size*0.1) //  FLAGS.batchsize
    
    models = []
    # with STRATEGY.scope(): BUGGY!
    models = buildModels(i, input_shape, train_data, val_data,steps_per_epoch,eval_steps_per_epoch)
    fitModels(models,train_data, val_data)
    del train_data
    del val_data
    del models
def main(argv):
  del argv  # Unused

  FLAGS.batchsize = FLAGS.batchsize #*STRATEGY.num_replicas_in_sync
  if not os.path.exists(FLAGS.outdir):
    os.system('mkdir '+FLAGS.outdir)

  extra = True #Use full training set
  test_data_list, train_data_list, val_data_list, info = getKfoldDataset(name="svhn_cropped",extra=extra) # Val data = 30220, Train data = 574168 , Test data = 26032
  nclasses    = info.features['label'].num_classes
  input_shape = info.features['image'].shape 
  if extra:
    train_size  = info.splits['train'].num_examples + info.splits['extra'].num_examples
  else:
    train_size = info.splits['train'].num_examples
    
  print("Using {}-fold training and validation data".format(len(train_data_list)))
  print("Training model")
  train(train_data_list, val_data_list, input_shape,train_size)
  print("Done!")

if __name__ == '__main__':

  GPUS = tf.config.experimental.list_physical_devices('GPU')
  if GPUS:
    
    # Create 2 virtual GPUs per GPU with ~4GB memory each
    try:
      # for gpu in GPUS:
        # tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096),tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(GPUS), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      print(e) 
  # STRATEGY = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()) BUGGY WITH PRUNING!
  
  OPTIMIZER   = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
  # LOSS        = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #When not targeting softmax but integers
  LOSS        = tf.keras.losses.CategoricalCrossentropy()
  
  
  app.run(main)
  