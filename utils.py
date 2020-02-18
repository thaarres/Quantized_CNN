from __future__ import print_function
import numpy as np
import sys, os, yaml
from optparse import OptionParser
import json

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
from qkeras import quantized_bits

print("Importing private libraries")
from callbacks import all_callbacks
import models


def parse_config(config_file) :

  print("Loading configuration from", config_file)
  config = open(config_file, 'r')
  return yaml.load(config)
  
def toJSON(model, outfile_name):
  outfile = open(outfile_name,'w')
  jsonString = model.to_json()
  with outfile:
    obj = json.loads(jsonString)
    json.dump(obj, outfile, sort_keys=True,indent=4, separators=(',', ': '))
    outfile.write('\n')
    
def rgb2gray(images):
    return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)

def formatSVHNArray(data):
    im = []
    for i in range(0, data.shape[3]):
        im.append(rgb2gray(data[:, :, :, i]))
    return np.asarray(im)

def fixSVHNLabel(labels):
    labels[labels == 10] = 0
    return labels

def plot_images(img, labels, nrows, ncols,outname='svhn_data.png'):
    fig, axes = plt.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flat): 
        if img[i].shape == (32, 32, 3):
            ax.imshow(img[i])
        else:
            ax.imshow(img[i,:,:,0])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(labels[i])
    fig.savefig(outname)    
    
def getDatasets(nclasses,mnist=False,svhn=False,greyScale=False):
  
  if svhn:
    mat_train = loadmat('train_32x32.mat', squeeze_me=True)     # 73257 +extra:531131
    mat_test  = loadmat('test_32x32.mat', squeeze_me=True)     # 26032
    X_train = mat_train['X']
    y_train = mat_train['y']
    X_test  = mat_test['X']
    y_test  = mat_test['y']
  
    X_train, X_test =  X_train.transpose((3,0,1,2)), X_test.transpose((3,0,1,2))
    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0
    
    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0
    Y_train = to_categorical(y_train, nclasses)
    Y_test  = to_categorical(y_test , nclasses)
    
    if greyScale:
      X_train = rgb2gray(X_train).astype(np.float32)
      X_test = rgb2gray(X_test).astype(np.float32)
      
    plot_images(X_train, y_train, 2, 8)

  else:
    
    if mnist:
      (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:    
      (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = np.expand_dims(X_train,axis=-1)
    X_test  = np.expand_dims(X_test,axis=-1)
    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')
    X_train /= 255
    X_test  /= 255
  
    Y_train = to_categorical(y_train, nclasses)
    Y_test  = to_categorical(y_test , nclasses)
    
  return X_train,X_test,Y_train,Y_test

def trainingDiagnostics(histories,outdir,filename='/learning_curve.png'):
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
  #ax1.set_yscale("log", nonposy='clip')
  #ax2.set_yscale("log", nonposy='clip')
  plt.legend([l1, l2],["Train (per fold)", "Test (per fold)"])
  plt.savefig(outdir+filename)

def performanceSummary(scores,outdir,outname='/performance_summary.png'):
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
	plt.savefig(outdir+outname)

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
        