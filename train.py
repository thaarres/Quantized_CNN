"""2D Conv
This code is loosely based on an example
from J. Brownlee at https://machinelearningmastery.com/
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
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adadelta
from tensorflow.keras.callbacks import LearningRateScheduler
print("Using TensorFlow version: {}".format(tensorflow.__version__))
print("Using Keras version: {}".format(tensorflow.keras.__version__))
import tensorflow.keras.backend as K
K.set_image_data_format('channels_first')
print("Forcing image data format to ".format(K.image_data_format()))

print("Importing helper libraries")
from tensorflow.keras.utils import to_categorical, plot_model
import h5py
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

print("Importing private libraries")
from callbacks import all_callbacks
import models
from qkeras import quantized_bits

# plot diagnostic learning curves
def diagnostics(histories,outdir):
  plt.clf()
  f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
  for i in range(len(histories)):
    if i == 0:
      l1, = ax1.plot(histories[i].history['loss']        , marker='o', linestyle='dashed', color='rosybrown',alpha=0.5+i/10)
      l2, = ax1.plot(histories[i].history['val_loss']    , marker='o', linestyle='dashed', color='orangered',alpha=0.5+i/10)
    else:
      ax1.plot(histories[i].history['loss']        , marker='o', linestyle='dashed', color='rosybrown',alpha=0.5+i/10)
      ax1.plot(histories[i].history['val_loss']    , marker='o', linestyle='dashed', color='orangered',alpha=0.5+i/10)
    ax2.plot(histories[i].history['accuracy']    , marker='o', linestyle='dashed', color='rosybrown',alpha=0.5+i/10)
    ax2.plot(histories[i].history['val_accuracy'], marker='o', linestyle='dashed', color='orangered',alpha=0.5+i/10)
  ax1.set_ylabel("Cross entropy loss")
  ax2.set_xlabel("Epoch")
  ax1.text(0.98, 0.98, 'k-Fold cross-validaton , k=%i'%len(histories), verticalalignment='top',horizontalalignment='right',transform=ax1.transAxes,color='slategray', fontsize=8)
  ax2.set_ylabel("Classification accuracy")
  ax1.set_yscale("log", nonposy='clip')
  #ax2.set_yscale("log", nonposy='clip')
  plt.legend([l1, l2],["Train (per fold)", "Test (per fold)"])
  plt.savefig(outdir+'/learning_curve.png')
  
# evaluate model with k-fold cross-validation
def evaluateModel(model,dataX, dataY,epochs,batch_size,callbacks, n_folds=5):
	scores, histories = list(), list()
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	for i,(train_ix, test_ix) in enumerate(kfold.split(dataX)):
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), verbose=0)#,callbacks=callbacks)
		_, acc = model.evaluate(testX, testY, verbose=0)
		print(' Accuracy after fold %i = > %.3f' % (i,acc * 100.0))
		scores.append(acc)
		histories.append(history)
	return scores, histories

def performanceSummary(scores,outdir):
	plt.clf()
	label = ('$<m>=%.3f$ $\sigma$=%.3f (k=%i)' % (np.mean(scores)*100, np.std(scores)*100, len(scores)))
	fig, ax = plt.subplots()
	bp1 = ax.boxplot(scores, bootstrap=1000, notch=True, patch_artist=True, boxprops=dict(facecolor="rosybrown"),medianprops=dict(color="orangered"),showfliers=False)
	ax.legend([bp1["boxes"][0]], [label], loc='upper right')
	ax.text(0.98, 0.98, 'k-Fold cross-validaton , k=%i'%len(scores), verticalalignment='top',horizontalalignment='right',transform=ax.transAxes,color='slategray', fontsize=8)
	
	plt.ylabel("Accuracy")
	labels = [item.get_text() for item in ax.get_xticklabels()]
	labels[0] = '$<32,16>$'
	ax.set_xticklabels(labels)
	plt.savefig(outdir+'/performance_summary.png')

def getDatasets(nclasses,useFashion=False):
    
  if useFashion:
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
  else:    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')
  X_train /= 255
  X_test /= 255
  
  Y_train = to_categorical(y_train, nclasses)
  Y_test  = to_categorical(y_test , nclasses)

  X_test  = np.expand_dims(X_test, axis=1)
  X_train = np.expand_dims(X_train, axis=1)
  return X_train,X_test,Y_train,Y_test

def toJSON(keras_model, outfile_name):
  outfile = open(outfile_name,'w')
  jsonString = keras_model.to_json()
  import json
  with outfile:
    obj = json.loads(jsonString)
    json.dump(obj, outfile, sort_keys=True,indent=4, separators=(',', ': '))
    outfile.write('\n')

def getModel(yamlConfig):
  Filters    = yamlConfig['Filters'].split(",")
  Kernel     = yamlConfig['Kernel' ].split(",")
  Strides    = yamlConfig['Strides'].split(",")
  Activation = yamlConfig['Activation']
  
  model = getattr(models, yamlConfig['KerasModel'])
  if 'quantized' in yamlConfig['KerasModel']:
    quantizer_conv  = quantized_bits( int(yamlConfig['cnn_bits']) , int(yamlConfig['cnn_integers']))
    quantizer_dense = quantized_bits( int(yamlConfig['dense_bits']), int(yamlConfig['dense_integers']))
    keras_model = model(Input(shape=input_shape),
                        nclasses  ,
                        Filters   ,
                        Kernel    ,
                        Strides   ,
                        Activation,
                        quantizer_dense,
                        quantizer_conv)
  else:
    keras_model = model(Input(shape=input_shape),
                        nclasses  ,
                        Filters   ,
                        Kernel    ,
                        Strides   ,
                        Activation)
    keras_model.summary()
    toJSON(keras_model,outdir + '/KERAS_model.json')
    keras_model.compile(loss=yamlConfig['KerasLoss'], optimizer=Adam(lr=0.0001, decay=0.000025), metrics=['accuracy'])
    return keras_model

def getCallbacks():
  callbacks=all_callbacks(stop_patience=1000,
  lr_factor=0.5,
  lr_patience=10,
  lr_epsilon=0.000001,
  lr_cooldown=2,
  lr_minimum=0.0000001,
  outputDir=outdir)
  callbacks = callbacks.callbacks
  return callbacks
  
def parse_config(config_file) :

  print("Loading configuration from", config_file)
  config = open(config_file, 'r')
  return yaml.load(config)

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option('-c','--config'   ,action='store',type='string',dest='config'   ,default='mnist.yml', help='yaml config file')
  parser.add_option('-f','--fashionMNIST',action='store_true', dest='fashionMNIST', default=True, help='Use fashion MNIST rather than MNIST')
  (options,args) = parser.parse_args()

  yamlConfig = parse_config(options.config)
  outdir = yamlConfig['OutputDir']
  if not os.path.exists(outdir): os.system('mkdir '+outdir)
  else: input("Warning: output directory exists. Press Enter to continue...")

  epochs    = yamlConfig['Epochs']
  batchsize = yamlConfig['Batchsize'] 
  kFolds    = yamlConfig['Folds']
  
  print("Getting datasets")
  X_train, X_test, Y_train, Y_test  = getDatasets(nclasses=10,useFashion=options.fashionMNIST)
  nclasses    = Y_train.shape[1]
  input_shape = X_train.shape[1:]
  
  print("Defining model")
  model = getModel(yamlConfig)
  callbacks = getCallbacks()
  
  print("Evaluating model")
  scores, histories = evaluateModel(model,X_train,Y_train,epochs,batchsize,callbacks,n_folds=kFolds)
  
  print("Plotting loss and accuracy")
  diagnostics(histories,outdir)
  print("Accuracy mean and spread")
  performanceSummary(scores,outdir)
