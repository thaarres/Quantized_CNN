"""Training of 2D quantized CNN (with kFold cross validation)
Author: Thea Aarrestad

Training of floating point precision and quantized deep convolutional neural networks
"""
from optparse import OptionParser
import numpy as np
np.random.seed(1337)  # for reproducibility
import sys, os, yaml
print("Set TF logging level to minimum (INFO and WARNING messages are not printed)")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tempfile

print("Importing TensorFlow")
import tensorflow as tf
tf.debugging.set_log_device_placement(False)
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
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
import tensorboard

ConstantSparsity = pruning_schedule.ConstantSparsity


print("Importing helper libraries")

import h5py
#from sklearn.model_selection import KFold,StratifiedShuffleSplit #Switch to tf.data
import matplotlib.pyplot as plt
from scipy.io import loadmat
from qkeras import quantized_bits
print("Importing private libraries")
import models
from utils import getDatasets,getKfoldDataset, toJSON, parse_config, trainingDiagnostics, performanceSummary,preprocess,print_model_sparsity

print("Spread jobs over multiple GPUs")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Create multiple virtual GPUs with 1GB memory each on each listed GPU
  try:
    for gpu in gpus:
      tf.config.experimental.set_virtual_device_configuration(gpu,
                                                              [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
                                                               tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)
strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

OPTIMIZER   = Adam(lr=0.0001, decay=0.000025)
LOSS        = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
BUFFER_SIZE = 1024 
NCLASSES    = 10


def getCallbacks():
  if os.path.exists(outdir+'/logs/'):
    os.system('rm -rf '+outdir+'/logs/')
  earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
  mcp_save_m    = ModelCheckpoint(outdir+'/bestModel.h5', save_best_only=True, monitor='val_loss', mode='auto')
  mcp_save_w    = ModelCheckpoint(outdir+'/bestWeights.h5', save_best_only=True,save_weights_only=True, monitor='val_loss', mode='auto')
  tensorboard   = tf.keras.callbacks.TensorBoard(log_dir=outdir+'/logs/', update_freq='batch')
  reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss')#, factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
  
  return [tensorboard,earlyStopping, mcp_save_m,mcp_save_w,reduce_lr_loss]

def buildAndTrain(yamlConfig, input_shape, train_data, val_data,steps_per_epoch,eval_steps_per_epoch,outdir,prune=True):
  
  #Get full model  
  model      = getModel(yamlConfig['KerasModel'],yamlConfig, input_shape)
  model._name = "full"

  #Get pruned models
  prune = True
  if prune == True:
     #Prune dense layers only
     pruning_schedule            = tfmot.sparsity.keras.PolynomialDecay( initial_sparsity=0.0, final_sparsity=0.5, begin_step=2000, end_step=4000)
     model_for_layerwise_pruning = getModel("float_cnn_densePrune"  ,yamlConfig, input_shape) 
     model_for_layerwise_pruning._name  = "layerwise_pruning"
   
     #Prune full model
     model_for_full_pruning      = tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=pruning_schedule)
     model_for_full_pruning._name  = "full_pruning"

     models = [model, model_for_layerwise_pruning, model_for_full_pruning]
  else:
    models = [model]
    
  histories, scores = list (), list ()
  for model in models:
    print("Training model: {} ".format(model.name))
    model.summary()
    callbacks = getCallbacks()
    if model.name.find("pruning")!=-1:
      print("Model sparsity: {} ".format(model.name))
      print_model_sparsity(model)
      callbacks = [ pruning_callbacks.UpdatePruningStep(), pruning_callbacks.PruningSummaries(log_dir=outdir+'/logs_%s/'%model.name, profile_batch=0)]
    print("Start training loop:\n\n")
    toJSON(model,outdir + '/model_%s.json'%model.name)
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=["accuracy"])
              
    history = model.fit(train_data,
                        epochs =  epochs, 
                        validation_data=val_data,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=eval_steps_per_epoch,
                        callbacks=callbacks,
                        verbose=1)                    

    val_score = model.evaluate(val_data)
    print('\n Test loss:', val_score[0])
    print('\n Test accuracy:', val_score[1])
    histories.append(history)
    scores.append(val_score)
  return histories, scores
    
    
    
      
def trainModel(yamlConfig, train_data_list, val_data_list, epochs, batch_size, nclasses, input_shape,train_size, outdir,prune=True):

  scores_, histories_ = list(), list() 
  for i,(val_,train_) in enumerate(zip(val_data_list, train_data_list)): 
    if i>3: break
    print("Working on fold: {}".format(i))
    train_data = train_.map(preprocess).shuffle(BUFFER_SIZE).batch(batch_size).repeat()
    val_data   = val_ .map(preprocess).batch(batch_size)
    
    # Read a single batch of examples from the training set and display shapes.
    for img_feature, label in train_data:
      break
    print('img_feature.shape (batch_size, image_height, image_width) =', img_feature.shape)
    print('label.shape (batch_size, number_of_labels) =', label.shape)
    
    steps_per_epoch      = int(train_size*0.9)  // batch_size #90% train, 10% validation in 10-fold xval
    eval_steps_per_epoch = int(train_size*0.1) // batch_size
    
    #
    # train_data = train_data.take(5000)
    # val_data   = val_data.take(2000)
    
    with strategy.scope():
      histories,scores = buildAndTrain(yamlConfig, input_shape, train_data, val_data,steps_per_epoch,eval_steps_per_epoch,outdir,prune=True)
      scores_.append(scores)
      histories_.append(histories)
      
  np.savez(outdir+"/scores", scores)  
 
  return scores_, histories_
  

  
def getModel(modelName, yamlConfig,input_shape):
  Filters    = yamlConfig['Filters'] .split(",")
  Kernel     = yamlConfig['Kernel' ] .split(",")
  Strides    = yamlConfig['Strides'] .split(",")
  Pooling    = yamlConfig['Pooling'] .split(",")
  Dropout    = yamlConfig['Dropout'].split(",")
  Activation = yamlConfig['Activation']
  
  # with strategy.scope():
    
  model = getattr(models, modelName)
  if 'quantized' in modelName:
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
  scores, histories = trainModel(yamlConfig, train_data_list, val_data_list, epochs, batchsize, nclasses, input_shape,train_size, outdir)

  print("Plotting loss and accuracy")
  trainingDiagnostics(histories,outdir)
  # print("Accuracy mean and spread")
  # performanceSummary(scores,outdir)
