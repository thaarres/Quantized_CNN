#!/usr/bin/env python3

"""Training of 2D quantized CNN (with kFold cross validation)
Author: Thea Aarrestad

Training of floating point precision and quantized deep convolutional neural networks
"""
import os
import setGPU
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

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping,History,ModelCheckpoint,TensorBoard,ReduceLROnPlateau,TerminateOnNaN,LearningRateScheduler

print("Using TensorFlow version: {}".format(tf.__version__))
print("Using Keras version: {}".format(tf.keras.__version__))
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
import tensorflow_model_optimization as tfmot

from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule, prune_registry
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorboard

import h5py

from qkeras import *
from qkeras.utils import model_quantize

import models
from utils.utils import getKfoldDataset, preprocess, print_model_sparsity
from qdictionaries import allQDictionaries
from utils.callbacks import all_callbacks
from utils.autoq import runAutoQKeras

flags.DEFINE_boolean('optimize'  , False, 'Run AutoQ')
flags.DEFINE_boolean('single'    , False, 'Train on one dataset (no kFold)')
flags.DEFINE_boolean('debug'     , False, 'Debug training')
flags.DEFINE_boolean('prune'     , False, 'Prune model')
flags.DEFINE_boolean('quantize'  , False, 'Quantize model')
flags.DEFINE_boolean('loadModel' , False, 'Load model')
flags.DEFINE_string ('lmodel'    , None , 'models/model_zenuity_autoq.h5')

# Training setting
flags.DEFINE_string ('outdir'    , None , 'Output directory')
flags.DEFINE_list   ('Filters'   , None , 'Filters   ')
flags.DEFINE_list   ('Neurons'   , None , 'Neurons    ')
flags.DEFINE_list   ('Kernel'    , None , 'Kernel    ')
flags.DEFINE_list   ('Strides'   , None , 'Strides   ')
flags.DEFINE_list   ('Pooling'   , None , 'Pooling   ')
flags.DEFINE_list   ('Dropout'   , None , 'Pooling   ')
flags.DEFINE_string ('Activation', "relu", 'Activation')
flags.DEFINE_string ('KerasModel', None, 'KerasModel')
flags.DEFINE_integer('folds'     , 10, 'folds', lower_bound=0)
flags.DEFINE_integer('start'     , 0 , 'start fold', lower_bound=0)
flags.DEFINE_integer('epochs'    , 100, 'Nepochs', lower_bound=1)
flags.DEFINE_integer('batchsize' , 512, 'batchsize', lower_bound=1)
flags.DEFINE_integer('buffersize',1024, 'buffersize', lower_bound=100)
flags.DEFINE_integer('nclasses'  ,10, 'nclasses', lower_bound=1)

#For optimizer
flags.DEFINE_float  ('lr'        , 3E-3, 'learning rate', lower_bound=0.)
flags.DEFINE_float  ('beta_1'    , 0.9, 'beta_1', lower_bound=0.)
flags.DEFINE_float  ('beta_2'    , 0.999, 'beta_1', lower_bound=0.)
flags.DEFINE_float  ('epsilon'   , 1e-07, 'epsilon', lower_bound=0.)

def trial(x,a):
  return a*x
def score(x,a):
  return a*x  
def checkLayerSize(m):
  for layer in m.layers:
    if layer.__class__.__name__ in ['Conv2D', 'Dense']:
        w = layer.get_weights()[0]
        layersize = np.prod(w.shape)
        print("{}: {}".format(layer.name,layersize)) # 0 = weights, 1 = biases
        if (layersize > 4096): # assuming that shape[0] is batch, i.e., 'None'
           print("Layer {} is too large ({}), are you sure you want to train?".format(layer.name,layersize))
           
#Moved
# def runAutoQ(STRATEGY,train_data, val_data, input_shape,train_size):
#   train_data = train_data.map(preprocess).shuffle(FLAGS.buffersize).batch(FLAGS.batchsize)#.repeat() #see https://www.tensorflow.org/guide/data
#   val_data   = val_data.map(preprocess).batch(FLAGS.batchsize)
#   model = getModel("full_autoQ",FLAGS.KerasModel, input_shape)
#   LOSS        = tf.keras.losses.CategoricalCrossentropy()
#   OPTIMIZER   = Adam(learning_rate=FLAGS.lr, beta_1=FLAGS.beta_1, beta_2=FLAGS.beta_2, epsilon=FLAGS.epsilon, amsgrad=False)
#   model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=["accuracy"])
#   model.summary()
#   runAutoQKeras(STRATEGY,train_data,val_data,model)

def pruneFunction(layer):
  # pruning_params = {'pruning_schedule': sparsity.ConstantSparsity(0.75, begin_step=2000, frequency=100)}
    pruning_params = {'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.40,final_sparsity=0.75, begin_step=1000, end_step=8000, frequency=100)}
    if isinstance(layer, tf.keras.layers.Conv2D):
      return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
    if isinstance(layer, tf.keras.layers.Dense) and layer.name!='output':
      return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)  
    return layer

def pruneModel(model):
  
  m_pruned = tf.keras.models.clone_model( model, clone_function=pruneFunction)
  m_pruned._name = 'pruned75_'+model.name
  return m_pruned
    
def build_config(i):
  
  qtemplate = {'default'  : {'kernel_quantizer' : None, 'bias_quantizer' : None},
               'QDense' : {'kernel_quantizer' : None, 'bias_quantizer' : None},
               'QConv2D' : {'kernel_quantizer' : None, 'bias_quantizer' : None},
               'QActivation' : {'relu' : None}}
             
  if i == 1:
    quantizer = 'binary(alpha=1)'
  elif i == 2:
    quantizer = 'ternary(alpha=1)'
  else:
    quantizer = 'quantized_bits({},0,alpha=1)'.format(i)
  act_quantizer='quantized_relu({},0)'.format(i)
  for type_key in ['default', 'QDense', 'QConv2D']:
      for q_key in ['kernel_quantizer', 'bias_quantizer']:
          qtemplate[type_key][q_key] = quantizer
  qtemplate['QActivation']['relu'] = act_quantizer
  return qtemplate

def getQuantizedModel(precision,model, weightfile):
  
  model.load_weights(weightfile)
  config = build_config(precision)
  custom_objects = {'BatchNormalization' : tf.keras.layers.BatchNormalization}
  qmodel = model_quantize(model, config, precision, custom_objects=custom_objects, transfer_weights=True) 
  qmodel._name = 'quantized_%i'%(precision)
  for layer in qmodel.layers:
      if hasattr(layer, "kernel_quantizer"):
          print(layer.name, "kernel:", str(layer.kernel_quantizer_internal), "bias:", str(layer.bias_quantizer_internal))
      elif hasattr(layer, "quantizer"):
          print(layer.name, "quantizer:", str(layer.quantizer))
  return qmodel
      
def getQuantizedFromBits(model, fold, bitwidth,weightfile):
    qmodel = getQuantizedModel(bitwidth,model,weightfile)
    qmodel._name = 'quant_%ibit_%i'%(bitwidth,fold)
    
    if FLAGS.prune == True:
      qmodel.load_weights(weightfile)  
      return pruneModel(qmodel)
    else:  
      return qmodel  
  
def getQuantizedFromMaps(full_model,fold,input_shape,full_model_path="one_hot_v2/full_0/saved_model.h5"):
  qmodels = []
  transferWeights = False
  try:
    model = tf.keras.models.load_model(full_model_path)
    transferWeights = True
  except:
    model = full_model
  for name, dict_ in allQDictionaries.items():
    # Workaround for deserialization from JSON (used by model_quantize) not
    # setting _USE_V2_BEHAVIOR=True thus using old V1 implementation
    custom_objects = {'BatchNormalization' : tf.keras.layers.BatchNormalization}
    qmodel = model_quantize(model, config, bitwidth, custom_objects=custom_objects, transfer_weights=transferWeights) 
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
                        FLAGS.Neurons   ,
                        FLAGS.Dropout   ,
                        FLAGS.Activation,
                        options)
  return model
  
def getCallbacks(outdir_):
  """Gets callbacks for training.
    Arguments:
      outdir_: Output directory
    Returns:
      list of callbacks
    """
  callbacks = [
      tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
      tf.keras.callbacks.ModelCheckpoint(filepath=outdir_+'/model_best.h5',monitor="val_loss",verbose=0,save_best_only=True), 
      tf.keras.callbacks.ModelCheckpoint(filepath=outdir_+'/weights_best.h5',monitor="val_loss",verbose=0,save_weights_only=True),
      tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=2, min_lr=1E-6)   
  ]                           
  return callbacks

def fitModel(model,train_data, val_data, test_data, stepsPerEpoch,evalStepsPerEpoch):
  """Runs Keras fit and saves model.
    Arguments:
      STRATEGY: Mirrored strategy
      models: list of models to train
      train_data: training data
      val_data: validation data  
    Returns:
      None
    """
    
  if not os.path.exists(FLAGS.outdir+'/%s/'%model.name):
    os.system('mkdir '+FLAGS.outdir+'/%s/'%model.name)
    
  callbacks = getCallbacks(FLAGS.outdir+'/%s/'%model.name)           
  if FLAGS.prune == True:
    callbacks.append(pruning_callbacks.UpdatePruningStep())
  
  start = time.time()
  LOSS        = tf.keras.losses.CategoricalCrossentropy()
  OPTIMIZER   = Adam(learning_rate=FLAGS.lr, beta_1=FLAGS.beta_1, beta_2=FLAGS.beta_2, epsilon=FLAGS.epsilon, amsgrad=True) 
  model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=["accuracy"])
  model.summary()
                                                                           
  history = model.fit(train_data,
                      epochs =  FLAGS.epochs,
                      validation_data = val_data,
                      callbacks = callbacks,
                      verbose=1)     
  model.load_weights(FLAGS.outdir+'/%s/weights_best.h5'%model.name)                                                       
  history_dict = history.history
  pd.DataFrame.from_dict(history.history).to_csv(FLAGS.outdir+'/%s/history_dict.csv'%model.name,index=False)
  test_score = model.evaluate(test_data)
  print("Done training model {}".format(model.name))
  print('\n Test loss:', test_score[0])
  print('\n Test accuracy:', test_score[1])
  np.savez(FLAGS.outdir+'/%s/scores'%model.name, test_score)  
  
  if FLAGS.prune == True:
    model_stripped = strip_pruning(model)
    model_stripped.save(FLAGS.outdir + '/%s/%s.h5'%(model.name,model.name)) 
  else:
    model.save(FLAGS.outdir + '/%s/%s.h5'%(model.name,model.name))   
  end = time.time()
  print('\n It took {} minutes to train!\n'.format( (end - start)/60.))
      
def train(STRATEGY,train_data_list, val_data_list, test_data, input_shape,train_size):
  
  test_data   = test_data .map(preprocess).batch(FLAGS.batchsize)
  for i,(val_,train_) in enumerate(zip(val_data_list, train_data_list)): 
    print("Working on fold: {}".format(i))
    if FLAGS.single and i > 0:
      break
    if i < FLAGS.start:
      continue
      
    #Augment for more stabil model
    datagen = ImageDataGenerator(rotation_range=8,
                                 zoom_range=[0.95, 1.05],
                                 height_shift_range=0.10,
                                 shear_range=0.15)
                                                 
    train_data = train_.map(preprocess).shuffle(FLAGS.buffersize).batch(FLAGS.batchsize)#.from_generator(datagen)#.repeat() #see https://www.tensorflow.org/guide/data
    val_data   = val_ .map(preprocess).batch(FLAGS.batchsize)
    
      
    for img_feature, label in train_data:
      break
    print(" --------------INPUT INFO --------------")
    print('INPUT img_feature.shape (batch_size, image_height, image_width) =', img_feature.shape)
    print('INPUT label.shape (batch_size, number_of_labels) =', label.shape)
    print(" --------------INPUT INFO --------------")
    
    steps_per_epoch      = int(train_size*0.9)  // FLAGS.batchsize #90% train, 10% validation in 10-fold xval
    eval_steps_per_epoch = int(train_size*0.1) //  FLAGS.batchsize
    
    with STRATEGY.scope():
      
      if FLAGS.loadModel:
        if FLAGS.prune:
          model = tf.keras.models.load_model('{}/{}_{}/model_best.h5'.format(FLAGS.outdir,FLAGS.lmodel.replace('/','').replace('.h5',''),i),custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation,'QBatchNormalization':QBatchNormalization,'trial':trial,'score':score})
          model._name = '{}_{}'.format(FLAGS.lmodel.replace('/','').replace('.h5',''),i)
          model = pruneModel(model)
        else:
          model = tf.keras.models.load_model('{}/{}'.format(FLAGS.outdir,FLAGS.lmodel),custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation,'QBatchNormalization':QBatchNormalization,'trial':trial,'score':score})
          # layers = [l for l in omodel.layers]
          # x = layers[0].output
          # for l in range(1, len(layers)):
          #   if isinstance(layers[l], tf.keras.layers.BatchNormalization):
          #     x = BatchNormalization()(x)
          #   elif l >= 18:
          #     x = QDense(64,kernel_quantizer=quantized_bits(4,0,1,alpha=1.0), bias_quantizer=quantized_bits(4,0,1,alpha=1.0),name='dense_%i'%1, use_bias=False)(x)
          #     x = BatchNormalization()(x)
          #     x = QActivation('quantized_relu(32,16)',name='dense_act_%i'%1)(x)
          #     x = Dense(10,name='output_dense')(x)
          #     x = Activation('softmax',name='output_softmax')(x)
          #     break
          #   else:
          #     x = layers[l](x)
          # model = Model(inputs=layers[0].input, outputs=x)
          model._name = '{}_{}'.format(FLAGS.lmodel.replace('/','').replace('.h5',''),i)
          
      else:  
        model = getModel("full_%i"%i,FLAGS.KerasModel, input_shape)
        checkLayerSize(model)
        if FLAGS.quantize:
            precision      = [16,14,12,10, 8,6,4,3,2,1]
            getWeightsFrom = [32,16,14,12,10,8,6,4,3,2] 
          
            # if i == 2:
            #   precision      = [2,1]
            #   getWeightsFrom = [3,2]
            
          
            for p,w in zip(precision,getWeightsFrom):
              if w == 32:
                weightFile=FLAGS.outdir+"/full_%s"%i+"/weights_best.h5"
              else:
                weightFile=FLAGS.outdir+'/quant_%ibit_%i'%(w,i)+"/weights_best.h5"
              if FLAGS.prune:
                weightFile=FLAGS.outdir+'/quant_%ibit_%i'%(p,i)+"/weights_best.h5"
              
              print("Setting weights from {}".format(weightFile))
              model = getModel("full_%i"%i,FLAGS.KerasModel, input_shape)
              model = getQuantizedFromBits(model,i,p,weightFile)
              fitModel(model,train_data, val_data, test_data, steps_per_epoch,eval_steps_per_epoch)
        else:
          if FLAGS.prune:
            model.load_weights(FLAGS.outdir+"/full_%i"%i+'/weights_best.h5')  
            model = pruneModel(model)
      
      fitModel(model,train_data, val_data, test_data, steps_per_epoch,eval_steps_per_epoch)
    
def main(argv):
  del argv  # Unused
  
  if FLAGS.debug is True:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"
    tf.get_logger().setLevel("INFO")
  else:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
    tf.get_logger().setLevel("ERROR")
  
  # Set GPUs
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      print(e)
      
  # Set mirrored strategy
  STRATEGY = tf.distribute.MirroredStrategy( cross_device_ops=tf.distribute.ReductionToOneDevice()) #,devices=[ "/gpu:1", "/gpu:2"] #If anyone else is hogging a GPU, this might fail!
    
  print('Scaling batch size with N devices in sync: {}'.format(STRATEGY.num_replicas_in_sync))
  FLAGS.batchsize = FLAGS.batchsize#*STRATEGY.num_replicas_in_sync
  
  if not os.path.exists(FLAGS.outdir):
    os.system('mkdir '+FLAGS.outdir)
  
  # Get training data
  extra = True #Use full training set
  if FLAGS.optimize or FLAGS.debug:
    extra = False
  test_data, train_data_list, val_data_list, info = getKfoldDataset(name="svhn_cropped",extra=extra) # Val data = 30220, Train data = 574168 , Test data = 26032
  nclasses    = info.features['label'].num_classes
  input_shape = info.features['image'].shape 
  if extra:
    train_size  = info.splits['train'].num_examples + info.splits['extra'].num_examples
  else:
    train_size = info.splits['train'].num_examples 
  print("Using {}-fold training and validation data".format(len(train_data_list)))
  
  # if FLAGS.optimize:
  #   runAutoQ(STRATEGY,train_data_list[0], val_data_list[0], input_shape,train_size)
  # else:
  train(STRATEGY,train_data_list, val_data_list, test_data, input_shape,train_size)
  print("Done!")

if __name__ == '__main__':
  
  print(tf.test.is_gpu_available())
  app.run(main)
  
