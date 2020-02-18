"""Training of 2D quantized CNN (with kFold cross validation)
Author: Thea Aarrestad

Training of floating point precision and quantized deep convolutional neural networks
"""

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import sys, os, yaml
from optparse import OptionParser

print("Importing TensorFlow")
import tensorflow
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adadelta,Nadam 
from tensorflow.keras.callbacks import Callback, EarlyStopping,History,ModelCheckpoint,TensorBoard,ReduceLROnPlateau,TerminateOnNaN
from tensorflow.keras.callbacks import LearningRateScheduler
print("Using TensorFlow version: {}".format(tensorflow.__version__))
print("Using Keras version: {}".format(tensorflow.keras.__version__))
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
print("Forcing image data format to ".format(K.image_data_format()))

print("Importing helper libraries")
from tensorflow.keras.utils import to_categorical, plot_model
import h5py
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy.io import loadmat

print("Importing private libraries")
from callbacks import all_callbacks
import models
from qkeras import quantized_bits
from utils import getDatasets, toJSON, parse_config, trainingDiagnostics, performanceSummary, getCallbacks

  
# Fit with k-fold cross-validation
def evaluateModel(yamlConfig,dataX, dataY,epochs,batch_size, n_folds,outdir):
  earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
  mcp_save_m = ModelCheckpoint(outdir+'/bestModel.h5', save_best_only=True, monitor='val_loss', mode='min')
  mcp_save_w = ModelCheckpoint(outdir+'/bestWeights.h5', save_best_only=True,save_weights_only=True, monitor='val_loss', mode='min')
  
  reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
  scores, histories = list(), list()
  kfold = KFold(n_folds, shuffle=True, random_state=1)
  for train_ix, test_ix in kfold.split(dataX):
    model = getModel(yamlConfig)
    trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), verbose=0,callbacks=[earlyStopping, mcp_save_m,mcp_save_w,reduce_lr_loss])
    _, acc = model.evaluate(testX, testY, verbose=0)
    print(' Accuracy after fold = > %.3f' % (acc * 100.0))
    scores.append(acc)
    histories.append(history)
	
  np.savez(outdir+"/scores", scores)
  np.savez(outdir+"/histories", histories)
  
  return scores, histories
  
def getModel(yamlConfig):
  Filters    = yamlConfig['Filters'].split(",")
  Kernel     = yamlConfig['Kernel' ].split(",")
  Strides    = yamlConfig['Strides'].split(",")
  Activation = yamlConfig['Activation']
  
  model = getattr(models, yamlConfig['KerasModel'])
  if 'quantized' in yamlConfig['KerasModel']:
    quantizer_conv  = quantized_bits( int(yamlConfig['cnn_bits']) , int(yamlConfig['cnn_integers']))
    quantizer_dense = quantized_bits( int(yamlConfig['dense_bits']), int(yamlConfig['dense_integers']))
    model = model(Input(shape=input_shape),
                        nclasses  ,
                        Filters   ,
                        Kernel    ,
                        Strides   ,
                        Activation,
                        quantizer_dense,
                        quantizer_conv)
  else:
    model = model(Input(shape=input_shape),
                        nclasses  ,
                        Filters   ,
                        Kernel    ,
                        Strides   ,
                        Activation)
  model.summary()
  toJSON(model,outdir + '/model.json')
  model.compile(loss=yamlConfig['KerasLoss'], optimizer=Nadam(lr=0.0001, decay=0.000025), metrics=['accuracy'])
  
  return model

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option('-o','--outdir'   ,action='store',type='string',dest='outdir'   ,default='', help='yaml config file')
  parser.add_option('-c','--config'   ,action='store',type='string',dest='config'   ,default='mnist.yml', help='yaml config file')
  parser.add_option('-s','--svhn',action='store_true', dest='svhn', default=False, help='Use SVHN')
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
  kFolds    = yamlConfig['Folds']
  
  print("Getting datasets")
  X_train, X_test, Y_train, Y_test  = getDatasets(nclasses=10,mnist=options.mnist,svhn=options.svhn)
  nclasses    = Y_train.shape[1]
  input_shape = X_train.shape[1:]
  
  print("Evaluating model")
  scores, histories = evaluateModel(yamlConfig,X_train,Y_train,epochs,batchsize,kFolds,outdir)
  
  print("Plotting loss and accuracy")
  trainingDiagnostics(histories,outdir)
  print("Accuracy mean and spread")
  performanceSummary(scores,outdir)
