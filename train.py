"""This is the summary line

This code is loosely based on the example
from J. Brownlee at
 https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
"""

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import sys, os, yaml
from optparse import OptionParser

print("Importing TensorFlow")
import tensorflow
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential
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

print("Importing private libraries")
from callbacks import all_callbacks
import models
from qkeras import quantized_bits

# plot diagnostic learning curves
def diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		pyplot.subplot(2, 1, 1)
		pyplot.title('Loss (crossentropy)')
		pyplot.plot(histories[i].history['loss'], color='blue', label='Training set')
		pyplot.plot(histories[i].history['val_loss'], color='orange', label='Test set')
		# plot accuracy
		pyplot.subplot(2, 1, 2)
		pyplot.title('Classification Accuracy')
		pyplot.plot(histories[i].history['accuracy'], color='blue', label='Training set')
		pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='Test set')
	pyplot.show()
  
# evaluate model with k-fold cross-validation
def evaluateModel(model,dataX, dataY,epochs,batch_size,callbacks, n_folds=5):
	scores, histories = list(), list()
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	for train_ix, test_ix in kfold.split(dataX):
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), verbose=0,callbacks=callbacks)
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		scores.append(acc)
		histories.append(history)
	return scores, histories

def performanceSummary(scores):
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	pyplot.boxplot(scores)
	pyplot.show()  

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
  

  epochs = yamlConfig['Epochs']
  batchsize = 32
  
  print("Getting datasets")
  X_train, X_test, Y_train, Y_test  = getDatasets(nclasses=10,useFashion=options.fashionMNIST)
  nclasses    = Y_train.shape[1]
  input_shape = X_train.shape[1:]
  
  print("Defining model")
  model = getModel(yamlConfig)
  
  print("Evaluating model")
  
  callbacks=all_callbacks(stop_patience=1000,
  lr_factor=0.5,
  lr_patience=10,
  lr_epsilon=0.000001,
  lr_cooldown=2,
  lr_minimum=0.0000001,
  outputDir=outdir)
  callbacks = callbacks.callbacks
  scores, histories = evaluateModel(model,X_train,Y_train,epochs,batchsize,callbacks,n_folds=5)
  
  print("Plotting loss and accuracy")
  diagnostics(histories)
  print("Accuracy mean and spread")
  summarize_performance(scores)
