#!/usr/bin/env python3

"""Training of 2D quantized CNN (with kFold cross validation)
Author: Thea Aarrestad

Training of floating point precision and quantized deep convolutional neural networks
"""
import os
# print("Set TF logging level to minimum (INFO and WARNING messages are not printed)")
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #This is buggy with absl, set manually with SET TF_CPP_MIN_LOG_LEVEL=3?

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
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule, prune_registry
from tensorflow_model_optimization.sparsity.keras import strip_pruning
import tensorboard
# PruneRegistry = prune_registry.PruneRegistry
# PruneRegistry.supports(tf.keras.layers.BatchNormalization)

import h5py

from qkeras import *
from qkeras.utils import model_quantize

import models
from utils import getKfoldDataset, toJSON, preprocess, print_model_sparsity
from qdictionaries import allQDictionaries
from callbacks import all_callbacks

flags.DEFINE_boolean('debug'  , False, 'Debug training')
#Which training to perform
flags.DEFINE_boolean('prune'     , False, 'Prune model (dense only and full model pruning)')
flags.DEFINE_boolean('quantize'  , False, 'Quantize model')

# Training setting
flags.DEFINE_string ('outdir'    , None , 'Output directory')
flags.DEFINE_list   ('Filters'   , None , 'Filters   ')
flags.DEFINE_list   ('Kernel'    , None , 'Kernel    ')
flags.DEFINE_list   ('Strides'   , None , 'Strides   ')
flags.DEFINE_list   ('Pooling'   , None , 'Pooling   ')
flags.DEFINE_list   ('Dropout'   , None , 'Pooling   ')
flags.DEFINE_string ('Activation', "relu", 'Activation')
flags.DEFINE_string ('KerasModel', None, 'KerasModel')
flags.DEFINE_integer('folds'     , 10, 'folds', lower_bound=1)
flags.DEFINE_integer('epochs'    , 100, 'Nepochs', lower_bound=1)
flags.DEFINE_integer('batchsize' , 512, 'batchsize', lower_bound=1)
flags.DEFINE_integer('buffersize',1024, 'buffersize', lower_bound=100)
flags.DEFINE_integer('nclasses'  ,10, 'nclasses', lower_bound=1)

#For optimizer
flags.DEFINE_float  ('lr'        , 0.001, 'learning rate', lower_bound=0.)
flags.DEFINE_float  ('beta_1'    , 0.9, 'beta_1', lower_bound=0.)
flags.DEFINE_float  ('beta_2'    , 0.999, 'beta_1', lower_bound=0.)
flags.DEFINE_float  ('epsilon'   , 1e-07, 'epsilon', lower_bound=0.)

def setWeights(model,full_model_path):
  full_model = tf.keras.models.load_model(full_model_path)
  for layerA,layerB in zip(model.layers,full_model.layers):
    layerA.set_weights(layerB.get_weights())

def getPrunedModels(full_models,fold,input_shape):
  pruned_models = []
  for full_model in full_models:
    pruning_params = {'pruning_schedule': sparsity.ConstantSparsity(0.75, begin_step=2000, frequency=100)}
    m_pruned = prune.prune_low_magnitude(full_model, **pruning_params)
    m_pruned._name = 'pruned_'+full_model.name
    pruned_models.append(m_pruned)
  # m_dense_pruned = getModel("dense_pruned_%i"%fold,"float_cnn_densePrune" , input_shape,pruning_params)
  # model_for_1L_pruning   = getModel("1L_pruning_%i"%fold  ,"float_cnn_1L_Prune"   , input_shape,pruning_params)
  # m_pruned = getModel("pruned_%i"%fold,"float_cnn_allPrune" , input_shape,pruning_params)

  # return [model_for_layerwise_pruning,model_for_full_pruning,model_for_1L_pruning]
  # return [m_pruned,m_dense_pruned]
  return pruned_models
  
def getQuantizedFromBits(model, fold, bitwidths=[4,6,8,10,12,16]):
  qmodels = []
  for bitwidth in bitwidths: 
    config = {
      "conv2d_1": {
          "kernel_quantizer": "quantized_bits(%i,0,alpha=1)"%bitwidth,
          "bias_quantizer": "quantized_bits(%i,0,alpha=1)"%bitwidth
      },
      "QConv2D": {
          "kernel_quantizer": "quantized_bits(%i,0,alpha=1)"%bitwidth,
          "bias_quantizer": "quantized_bits(%i,0,alpha=1)"%bitwidth
      },
      "QDense": {
          "kernel_quantizer": "quantized_bits(%i,0,alpha=1)"%bitwidth,
          "bias_quantizer": "quantized_bits(%i,0,alpha=1)"%bitwidth
      },
      "QActivation": { "relu": "quantized_relu(%i)"%32 },
      "act_2": "quantized_relu(%i)"%32,
    }
    
    # q_dict_generic = {
#                     "conv_0": {
#                     "kernel_quantizer": "quantized_bits(%i,0,alpha=1)"%bitwidth,
#                     "bias_quantizer"  : "quantized_bits(%i,0,alpha=1)"%bitwidth
#                     },
#                     # "conv_1": {
#                     # "kernel_quantizer": "quantized_bits(%i,0,alpha=1)"%bitwidth,
#                     # "bias_quantizer"  : "quantized_bits(%i,0,alpha=1)"%bitwidth
#                     # },
#                     # "conv_2": {
#                     # "kernel_quantizer": "quantized_bits(%i,0,alpha=1)"%bitwidth,
#                     # "bias_quantizer"  : "quantized_bits(%i,0,alpha=1)"%bitwidth
#                     # },
#                     "dense_1": {
#                         "kernel_quantizer": "quantized_bits(%i,0,alpha=1)"%bitwidth,
#                         "bias_quantizer"  : "quantized_bits(%i,0,alpha=1)"%bitwidth
#                     }}
    qmodel = model_quantize(model, config, bitwidth, transfer_weights=False)
    qmodel._name = 'quant_%ibit_%i'%(bitwidth,fold)
    qmodels.append(qmodel)
    
    for layer in qmodel.layers:
        if hasattr(layer, "kernel_quantizer"):
            print(layer.name, "kernel:", str(layer.kernel_quantizer_internal), "bias:", str(layer.bias_quantizer_internal))
        elif hasattr(layer, "quantizer"):
            print(layer.name, "quantizer:", str(layer.quantizer))
       
  return qmodels  
  
def getQuantizedFromMaps(full_model,fold,input_shape,full_model_path="one_hot_v2/full_0/saved_model.h5"):
  qmodels = []
  transferWeights = False
  try:
    model = tf.keras.models.load_model(full_model_path)
    transferWeights = True
  except:
    model = full_model
  model.summary()
  for name, dict_ in allQDictionaries.items():
    qmodel = model_quantize(model, dict_, 4, transfer_weights=transferWeights)   
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
  callbacks = all_callbacks(stop_patience = 7,
                            lr_factor = 0.5,
                            lr_patience = 4,
                            lr_epsilon = 0.000001,
                            lr_cooldown = 2,
                            lr_minimum = 0.0000001,
                            outputDir = outdir_,
                            debug = 0)
  return callbacks

def fitModels(models,train_data, val_data):
  for model in models:
    callbacks = getCallbacks(FLAGS.outdir+'/%s/'%model.name)           
    
    if FLAGS.prune == True:
      callbacks.callbacks.append(pruning_callbacks.UpdatePruningStep())
    
    start = time.time()
    print("Training model {}".format(model.name))
    # model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=["accuracy"])
    history = model.fit(train_data,
                        epochs =  FLAGS.epochs, 
                        validation_data = val_data,
                        callbacks = callbacks.callbacks,
                        verbose=1)     
    model.load_weights(FLAGS.outdir+'/%s/KERAS_check_best_model_weights.h5'%model.name)                                                       
    history_dict = history.history
    pd.DataFrame.from_dict(history.history).to_csv(FLAGS.outdir+'/%s/history_dict.csv'%model.name,index=False)
    val_score = model.evaluate(val_data)
    print("Done training model {}".format(model.name))
    print('\n Test loss:', val_score[0])
    print('\n Test accuracy:', val_score[1])
    np.savez(FLAGS.outdir+'/%s/scores'%model.name, val_score)
    toJSON(model,FLAGS.outdir + '/%s/model.json'%model.name)   
    model.save(FLAGS.outdir + '/%s/tf_saved_model.h5'%model.name, save_format='tf') 
    if FLAGS.prune == True:
      model_stripped = strip_pruning(model)
      model.save(FLAGS.outdir + '/%s/tf_saved_model_stripped.h5'%model.name, save_format='tf')
    del model
    end = time.time()
    print('\n It took {} minutes to train!\n'.format( (end - start)/60.))
  
    
def buildModels(fold, input_shape, train_data, val_data,steps_per_epoch,eval_steps_per_epoch):
  
  models = [] 
  full_model = getModel("full_%i"%fold,FLAGS.KerasModel, input_shape)
  
  if FLAGS.quantize == True:
    # models = getQuantizedFromMaps(full_model,fold,input_shape)
    qmodels = getQuantizedFromBits(full_model,fold,[4])
    models  = qmodels
    if FLAGS.prune == True:
      pqmodels = getPrunedModels(qmodels,fold,input_shape)
      models = qmodels+pqmodels  
  elif FLAGS.prune == True:
    models = getPrunedModels([full_model],fold,input_shape)
    try:
      full_model_path="one_hot_v2/full_0/saved_model.h5"
      model = tf.keras.models.load_model(full_model_path)
      for prunedModel in models:
        setWeights(prunedModel,model)
    except:
      print(" No pretrained model found! Initialising pruned model weights randomly")
      
  else:
    models = [full_model]

  for i,model in enumerate(models):
    
    
    
    if not os.path.exists(FLAGS.outdir+'/%s/'%model.name):
      os.system('mkdir '+FLAGS.outdir+'/%s/'%model.name)
    
    # LOSS        = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #When not targeting softmax but integers
    LOSS        = tf.keras.losses.CategoricalCrossentropy()
    OPTIMIZER   = Adam(learning_rate=FLAGS.lr, beta_1=FLAGS.beta_1, beta_2=FLAGS.beta_2, epsilon=FLAGS.epsilon, amsgrad=False) 
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=["accuracy"])
    model.summary()
    if FLAGS.quantize == True:
      print_qstats(model)
  return models          
    
      
def train(STRATEGY,train_data_list, val_data_list, input_shape,train_size):
  
  for i,(val_,train_) in enumerate(zip(val_data_list, train_data_list)): 
    print("Working on fold: {}".format(i))
    if i > 4: break
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
    print('Spreading weights over N={} GPUs and updating in sync. Bulding models:'.format(STRATEGY.num_replicas_in_sync))
    with STRATEGY.scope(): #must contain: creation of Keras model, optimizer and metrics
      models = buildModels(i, input_shape, train_data, val_data,steps_per_epoch,eval_steps_per_epoch)
    fitModels(models,train_data, val_data)
    del train_data
    del val_data
    del models
    
def main(argv):
  del argv  # Unused
  
  if FLAGS.debug is True:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"
    tf.get_logger().setLevel("INFO")
  else:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
    tf.get_logger().setLevel("ERROR")
    
  print("----------Setting up GPUs----------")
  
  # Set memory growth
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        #tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096),tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      print(e)
      
  # Set mirrored strategy
  STRATEGY = tf.distribute.MirroredStrategy( cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()) #,devices=[ "/gpu:1", "/gpu:2"] #If anyone else is hogging a GPU, this might fail!
  
  print('Scaling batch size with N devices in sync: {}'.format(STRATEGY.num_replicas_in_sync))
  FLAGS.batchsize = FLAGS.batchsize*STRATEGY.num_replicas_in_sync
  
  if not os.path.exists(FLAGS.outdir):
    os.system('mkdir '+FLAGS.outdir)
  
  # Get training data
  extra = True #Use full training set
  test_data_list, train_data_list, val_data_list, info = getKfoldDataset(name="svhn_cropped",extra=extra) # Val data = 30220, Train data = 574168 , Test data = 26032
  nclasses    = info.features['label'].num_classes
  input_shape = info.features['image'].shape 
  if extra:
    train_size  = info.splits['train'].num_examples + info.splits['extra'].num_examples
  else:
    train_size = info.splits['train'].num_examples
  
  # Train!  
  print("Using {}-fold training and validation data".format(len(train_data_list)))
  train(STRATEGY,train_data_list, val_data_list, input_shape,train_size)
  print("Done!")

if __name__ == '__main__':
  
  app.run(main)
  