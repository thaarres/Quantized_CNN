"""Training of 2D quantized CNN (with kFold cross validation)
Author: Thea Aarrestad

Training of floating point precision and quantized deep convolutional neural networks
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from optparse import OptionParser
import numpy as np
np.random.seed(1337)  # for reproducibility
import sys, os, yaml

print("Importing TensorFlow")
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adadelta,Nadam 
from tensorflow.keras.callbacks import Callback, EarlyStopping,History,ModelCheckpoint,TensorBoard,ReduceLROnPlateau,TerminateOnNaN,LearningRateScheduler
from tensorflow.keras.utils import to_categorical, plot_model

print("Using TensorFlow version: {}".format(tf.__version__))
print("Using Keras version: {}".format(tf.keras.__version__))
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
print("Forcing image data format to {}".format(K.image_data_format()))
import tensorflow_model_optimization as tfmot

print("Importing helper libraries")

import h5py
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from scipy.io import loadmat
from qkeras import quantized_bits
print("Importing private libraries")
import models
from utils import getDatasets,getKfoldDataset, toJSON, parse_config, trainingDiagnostics, performanceSummary,preprocess

print("Limit GPU usage")
#import setGPU
#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
print("Spread jobs over multiple GPUs")
# strategy = tf.distribute.MirroredStrategy()

# config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.35
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

OPTIMIZER   = Adam(lr=0.0001, decay=0.000025)
LOSS        = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
BUFFER_SIZE = 10000 # Use a much larger value for real code.
NCLASSES    = 10


def getCallbacks(name='full'):
  earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
  mcp_save_m    = ModelCheckpoint(outdir+'/bestModel_%s.h5'  %name, save_best_only=True, monitor='val_loss', mode='auto')
  mcp_save_w    = ModelCheckpoint(outdir+'/bestWeights_%s.h5'%name, save_best_only=True,save_weights_only=True, monitor='val_loss', mode='auto')
  tensorboard   = tf.keras.callbacks.TensorBoard(log_dir='./logs/model_%s'%name, update_freq='batch')
  reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss')#, factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
  
  return [earlyStopping, mcp_save_m,mcp_save_w,reduce_lr_loss]


def evaluateModel(yamlConfig, train_data_list, val_data_list, epochs, batch_size, nclasses, input_shape,train_size, outdir):
  
  # BATCH_SIZE_PER_REPLICA = batch_size
  # BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
  
  scores, histories = list(), list()  
  for i,(val_,train_) in enumerate(zip(val_data_list, train_data_list)): 
    train_data = train_.map(preprocess).shuffle(BUFFER_SIZE).batch(batch_size).repeat()
    val_data   = val_ .map(preprocess).batch(batch_size)
    
    # Read a single batch of examples from the training set and display shapes.
    for img_feature, label in train_data:
      break
    print('img_feature.shape (batch_size, image_height, image_width) =', img_feature.shape)
    print('label.shape (batch_size, number_of_labels) =', label.shape)
    
    steps_per_epoch      = int(train_size*0.9)  // batch_size #90% train, 10% validation in 10-fold xval
    eval_steps_per_epoch = int(train_size*0.1) // batch_size
      
    
  
    model = getModel(yamlConfig, input_shape)
    
    #Add pruned model
    prune = True
    if prune == True:
      pruning_schedule  = tfmot.sparsity.keras.PolynomialDecay( initial_sparsity=0.0, final_sparsity=0.5, begin_step=2000, end_step=4000)
      model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=pruning_schedule)
      model_for_pruning.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=["accuracy"])
      model_for_pruning.fit(train_data,
                            epochs =  epochs, 
                            validation_data=val_data,
                            steps_per_epoch=steps_per_epoch,
                            validation_steps=eval_steps_per_epoch,
                            callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
                            verbose=1)
                            
    allCallbacks_full   = getCallbacks(name="full")
   
    history = model.fit(train_data,
                        epochs =  epochs, 
                        validation_data=val_data,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=eval_steps_per_epoch,
                        callbacks=allCallbacks,
                        verbose=1)
    
    loss, acc = model.evaluate(val_data)
    print("Loss {}, Val accuracy {}".format(loss, acc ))  
    print("Loss {}, Val accuracy {}".format(loss, acc ))  
    print("Loss {}, Val accuracy {}".format(loss, acc ))  
    model.save(outdir+'finalModel_%i.h5'%i, save_format='tf')
    scores.append(acc)
    histories.append(history)
  np.savez(outdir+"/scores", scores)  
 
  return scores, histories
  

  
def getModel(yamlConfig,input_shape):
  Filters    = yamlConfig['Filters'] .split(",")
  Kernel     = yamlConfig['Kernel' ] .split(",")
  Strides    = yamlConfig['Strides'] .split(",")
  Pooling    = yamlConfig['Pooling'] .split(",")
  Dropout    = yamlConfig['Dropout'].split(",")
  Activation = yamlConfig['Activation']
  
  # with strategy.scope():
    
  model = getattr(models, yamlConfig['KerasModel'])
  if 'quantized' in yamlConfig['KerasModel']:
    quantizer_conv  = quantized_bits( int(yamlConfig['cnn_bits']) , int(yamlConfig['cnn_integers']), 1)
    quantizer_dense = quantized_bits( int(yamlConfig['dense_bits']), int(yamlConfig['dense_integers']), 1)
    model = model(Input(input_shape),
                        NCLASSES  ,
                        Filters   ,
                        Kernel    ,
                        Strides   ,
                        Pooling   ,
                        Dropout   ,
                        Activation,
                        quantizer_dense,
                        quantizer_conv)
  else:
    model = model(Input(input_shape),
                        NCLASSES  ,
                        Filters   ,
                        Kernel    ,
                        Strides   ,
                        Pooling   ,
                        Dropout   ,
                        Activation)
  model.summary()
  toJSON(model,outdir + '/model.json')
  model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=["accuracy"])
  
  return model


if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option('-o','--outdir'   ,action='store',type='string',dest='outdir'   ,default='', help='yaml config file')
  parser.add_option('-c','--config'   ,action='store',type='string',dest='config'   ,default='float_cnn.yml', help='yaml config file')
  parser.add_option('-s','--svhn',action='store_true', dest='svhn', default=True, help='Use SVHN')
  parser.add_option('--mnist',action='store_true', dest='mnist', default=False, help='Use MNIST')
  (options,args) = parser.parse_args()

  yamlConfig = parse_config(options.config)
  if not options.outdir:
    outdir = yamlConfig['OutputDir']
  else:
    outdir = options.outdir
    
  if not os.path.exists(outdir): os.system('mkdir '+outdir)
  else: input("Warning: output directory exists. Press Enter to continue...")

  epochs    = yamlConfig['Epochs']
  batchsize = yamlConfig['Batchsize'] 


  extra = True
  test_data_list, train_data_list, val_data_list, info = getKfoldDataset(name="svhn_cropped",extra=extra) # Val data = 30220, Train data = 574168 , Test data = 26032
  nclasses    = info.features['label'].num_classes
  input_shape = info.features['image'].shape 
  if extra:
    train_size  = info.splits['train'].num_examples + info.splits['extra'].num_examples
  else:
    train_size = info.splits['train'].num_examples
    
  print("Using {}-fold training and validation data".format(len(train_data_list)))
  print("Evaluating model")
  scores, histories = evaluateModel(yamlConfig, train_data_list, val_data_list, epochs, batchsize, nclasses, input_shape,train_size, outdir)

  print("Plotting loss and accuracy")
  trainingDiagnostics(histories,outdir)
  # print("Accuracy mean and spread")
  # performanceSummary(scores,outdir)
