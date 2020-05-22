from __future__ import print_function
import numpy as np
import sys, os, yaml
from optparse import OptionParser
import json
import cv2
from matplotlib.pyplot import imread 
import matplotlib.patches as mpatch
from matplotlib.patches import FancyBboxPatch
print("Importing TensorFlow")
import tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds
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
#import tensorflow_datasets as tfds
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper


print("Importing helper libraries")
from tensorflow.keras.utils import to_categorical, plot_model
import h5py
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy.io import loadmat
from qkeras import quantized_bits
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
sns.set(style="whitegrid")
            
print("Importing private libraries")
from callbacks import all_callbacks
import models

def getFeaturesLabels(name="svhn_cropped"):
  (img_train, label_train), (img_test, label_test) = tfds.as_numpy(tfds.load(name, split=['train', 'test'], batch_size=-1, as_supervised=True,))
  return (img_train, label_train),(img_test, label_test)
  
def dataset2numpy(dataset, steps=1):
    "Helper function to get data/labels back from TF dataset"
    iterator = dataset.make_one_shot_iterator()
    next_val = iterator.get_next()
    with tf.Session() as sess:
        for _ in range(steps):
           inputs, labels = sess.run(next_val)
           yield inputs, labels
           
def parse_config(config_file) :

  print("Loading configuration from", config_file)
  config = open(config_file, 'r')
  return yaml.safe_load(config)
  
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

def plot_images(img, labels, nrows, ncols,outname='svhn_data.pdf'):
    fig, axes = plt.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flat): 
        if img[i].shape == (32, 32, 3):
            ax.imshow(img[i])
        else:
            ax.imshow(img[i,:,:,0])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(labels[i])
    fig.savefig(outname)    
    
def getDatasets(nclasses,doMnist=False,doSvhn=False,greyScale=False,ext=False):
  
  if doSvhn:
    
    mat_train = loadmat('train_32x32.mat', squeeze_me=True)     # 73257 +extra:531131
    mat_test  = loadmat('test_32x32.mat', squeeze_me=True)     # 26032
    
    if ext:
      mat_train_ext = loadmat('extra_32x32.mat', squeeze_me=True)
      x_train = np.concatenate((mat_train['X'] , mat_train_ext['X']), axis=-1)
      y_train = np.concatenate((mat_train['y'] , mat_train_ext['y']))
    else:
      x_train = mat_train['X']
      y_train = mat_train['y']
      
    x_test  = mat_test['X']
    y_test  = mat_test['y']
  
    x_train, x_test =  x_train.transpose((3,0,1,2)), x_test.transpose((3,0,1,2))
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0
    
    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0
    # y_train = to_categorical(y_train, nclasses)
    # y_test  = to_categorical(y_test , nclasses)
    
    if greyScale:
      x_train = rgb2gray(x_train).astype(np.float32)
      x_test  = rgb2gray(x_test).astype(np.float32)
      
    #plot_images(X_train, y_train, 2, 8)

  else:
    
    if doMnist:
      (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:    
      (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_test_orig = x_test

    x_train = x_train.astype("float32")
    x_test  = x_test.astype("float32")

    x_train = x_train[..., np.newaxis]
    x_test  = x_test[..., np.newaxis]

    x_train /= 255.0
    x_test /= 255.0

    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    print(y_train[0:10])

    # y_train = to_categorical(y_train, nclasses)
    # y_test  = to_categorical(y_test, nclasses)
  
  return x_train,x_test,y_train,y_test

def preprocess(image, label,nclasses=10):
  image = tf.cast(image, tf.float32) / 255.
  label = tf.one_hot(tf.squeeze(label), nclasses)
  return image, label
  
def getKfoldDataset(name="svhn_cropped",extra=False,val_percent=10):    
  # Construct a tf.data.Dataset
  # dataset, info  = tfds.load(name=name, with_info=True, as_supervised=True)
  # train_, test_, extra_ = dataset['train'], dataset['test'], dataset['extra']
  # test_data = tfds.load(name, split=[f'test[:{k}%]+test[{k+20}%:]+test[:{k}%]+test[{k+20}%:]'for k in range(0, 100, 20)], as_supervised=True)
  test_data = tfds.load(name, split='test', as_supervised=True)
  if extra:
      val_data         = tfds.load(name, split=[f'train[{k}%:{k+10}%]+extra[{k}%:{k+10}%]'for k in range(0, 100, 10)], with_info=False, as_supervised=True,shuffle_files=True)
      train_data, info = tfds.load(name, split=[f'train[:{k}%]+train[{k+10}%:]+extra[:{k}%]+extra[{k+10}%:]'for k in range(0, 100, 10)], with_info=True, as_supervised=True,shuffle_files=True)
  else:                           
    val_data         = tfds.load(name, split=[f'train[{k}%:{k+10}%]'for k in range(0, 100, 10)], with_info=False, as_supervised=True,shuffle_files=True)
    train_data, info = tfds.load(name, split=[f'train[:{k}%]+train[{k+10}%:]'for k in range(0, 100, 10)], with_info=True, as_supervised=True,shuffle_files=True) 
  
  return test_data, train_data, val_data, info
  
def trainingDiagnostics(historiesPerFold,outdir,filename='learning_curve.pdf'):
  print("Plotting loss and accuracy per epoch for {} folds".format(len(historiesPerFold)))
  plt.clf()
  f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
  for i,model in enumerate(historiesPerFold):
    if i == 0:
      l1, = ax1.plot(model['loss']        , marker='o', linestyle='dashed', color='rosybrown',alpha=0.5+i/10)
      l2, = ax1.plot(model['val_loss']    , marker='o', linestyle='dashed', color='orangered',alpha=0.5+i/10)
    else:
      ax1.plot(model['loss']        , marker='o', linestyle='dashed', color='rosybrown',alpha=0.5+i/10)
      ax1.plot(model['val_loss']    , marker='o', linestyle='dashed', color='orangered',alpha=0.5+i/10)
    ax2.plot(1-model['accuracy']    , marker='o', linestyle='dashed', color='rosybrown',alpha=0.5+i/10)
    ax2.plot(1-model['val_accuracy'], marker='o', linestyle='dashed', color='orangered',alpha=0.5+i/10)
  ax1.set_ylabel("Sparse Cross ent. loss")
  ax1.set_yscale("log")
  ax1.text(0.98, 0.98, 'k-Fold cross-validaton , k=%i'%len(historiesPerFold), verticalalignment='top',horizontalalignment='right',transform=ax1.transAxes,color='slategray', fontsize=8)
  ax2.set_xlabel("Epoch")
  ax2.set_ylabel("1-Classification acc.")
  ax2.set_yscale("log")
  #ax1.set_ypreprocess("log", nonposy='clip')
  #ax2.set_ypreprocess("log", nonposy='clip')
  plt.legend([l1, l2],["Train (per fold)", "Test (per fold)"])
  for axis in [ax1.xaxis, ax1.yaxis,ax2.xaxis, ax2.yaxis]:
    scalarFormatter = ScalarFormatter()
    scalarFormatter.set_scientific(False)
    axis.set_major_formatter(scalarFormatter)
    axis.set_minor_formatter(scalarFormatter)
    axis.set_minor_locator(plt.MaxNLocator(3))
    axis.set_major_locator(plt.MaxNLocator(3))
  plt.savefig(outdir+"/"+filename)

def performanceSummary(scores,labels, outdir,outname='/performance_summary.pdf',folds=10):
  plt.clf()
  fig, ax = plt.subplots()
  # plt.legend(loc='upper left',fontsize=15)
  plt.grid(color='0.8', linestyle='dotted')
  # plt.figtext(0.925, 0.94,m.replace('_',' '), wrap=True, horizontalalignment='right')
  add_logo(ax, fig, 0.14, position='upper right')
  bins = np.linspace(-1.5, 1.5, 50)
  colors = sns.color_palette("colorblind", len(scores))
  for i, model in enumerate(scores):
    bp1 = ax.boxplot(model, positions=[i], bootstrap=1000, notch=False, widths=0.5, patch_artist=True, boxprops=dict(facecolor=colors[i]),medianprops=dict(color="black"),showfliers=True)
  
  ax.set_ylim(0.1,1.0)
  # plt.legend(loc='upper left',fontsize=15)
	
	# for i,(score,label) in enumerate(zip(scores,labels)):
#     boxes.append = ( ax.boxplot(score, bootstrap=1000, notch=True, patch_artist=True, boxprops=dict(facecolor="rosybrown"),medianprops=dict(color="orangered"),showfliers=False,positions=[i],widths=0.5),label=)
#   print(boxes[i])
#   ax.legend([boxes['boxes'][0]], [labels], loc='upper right')
  ax.text(0.1 , 0.98, 'k-Fold cross-validaton , k=%i'%folds, verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,color='slategray', fontsize=8)
#
  plt.ylabel("Accuracy")
  labels_ = [item.get_text() for item in ax.get_xticklabels()]
  for i in range(0,len(labels)):
    labels_[i] = labels[i]
  ax.set_xticklabels(labels_)
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

def add_logo(ax, fig, zoom, position='upper left', offsetx=10, offsety=10, figunits=False):

  #resize image and save to new file
  img = cv2.imread('logo.jpg', cv2.IMREAD_UNCHANGED)
  im_w = int(img.shape[1] * zoom )
  im_h = int(img.shape[0] * zoom )
  dim = (im_w, im_h)
  resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  cv2.imwrite( "logo_resized.jpg", resized );

  #read resized image
  im = imread('logo_resized.jpg')
  im_w = im.shape[1]
  im_h = im.shape[0]

  #get coordinates of corners in data units and compute an offset in pixel units depending on chosen position
  ax_xmin,ax_ymax = 0,0
  offsetX = 0
  offsetY = 0
  if position=='upper left':
   ax_xmin,ax_ymax = ax.get_xlim()[0],ax.get_ylim()[1]
   offsetX,offsetY = offsetx,-im_h-offsety
  elif position=='out left':
   ax_xmin,ax_ymax = ax.get_xlim()[0],ax.get_ylim()[1]
   offsetX,offsetY = offsetx,offsety
  elif position=='upper right':
   ax_xmin,ax_ymax = ax.get_xlim()[1],ax.get_ylim()[1]
   offsetX,offsetY = -im_w-offsetx,-im_h-offsety
  elif position=='out right':
   ax_xmin,ax_ymax = ax.get_xlim()[1],ax.get_ylim()[1]
   offsetX,offsetY = -im_w-offsetx,offsety
  elif position=='bottom left':
   ax_xmin,ax_ymax = ax.get_xlim()[0],ax.get_ylim()[0]
   offsetX,offsetY=offsetx,offsety
  elif position=='bottom right':
   ax_xmin,ax_ymax = ax.get_xlim()[1],ax.get_ylim()[0]
   offsetX,offsetY=-im_w-offsetx,offsety
         
  #transform axis limits in pixel units
  ax_xmin,ax_ymax = ax.transData.transform((ax_xmin,ax_ymax))
  #compute figure x,y of bottom left corner by adding offset to axis limits (pixel units)
  f_xmin,f_ymin = ax_xmin+offsetX,ax_ymax+offsetY
       
  #add image to the plot
  plt.figimage(im,f_xmin,f_ymin)
       
  #compute box x,y of bottom left corner (= figure x,y) in axis/figure units
  if figunits:
   b_xmin,b_ymin = fig.transFigure.inverted().transform((f_xmin,f_ymin))
   #print("figunits",b_xmin,b_ymin)
  else: b_xmin,b_ymin = ax.transAxes.inverted().transform((f_xmin,f_ymin))
  
  #compute figure width/height in axis/figure units
  if figunits: f_xmax,f_ymax = fig.transFigure.inverted().transform((f_xmin+im_w,f_ymin+im_h)) #transform to figure units the figure limits
  else: f_xmax,f_ymax = ax.transAxes.inverted().transform((f_xmin+im_w,f_ymin+im_h)) #transform to axis units the figure limits
  b_w = f_xmax-b_xmin
  b_h = f_ymax-b_ymin

  #set which units will be used for the box
  transformation = ax.transAxes
  if figunits: transformation=fig.transFigure
  
  rectangle = FancyBboxPatch((b_xmin,b_ymin),
                              b_w, b_h,
                  transform=transformation,
                  boxstyle='round,pad=0.004,rounding_size=0.01',
                  facecolor='w',
                  edgecolor='0.8',linewidth=0.8,clip_on=False)
  ax.add_patch(rectangle)
  
def print_model_sparsity(pruned_model):
  """Prints sparsity for the pruned layers in the model.
  Model Sparsity Summary
  --
  prune_lstm_1: (kernel, 0.5), (recurrent_kernel, 0.6)
  prune_dense_1: (kernel, 0.5)
  Args:
    pruned_model: keras model to summarize.
  Returns:
    None
  """
  def _get_sparsity(weights):
    return 1.0 - np.count_nonzero(weights) / float(weights.size)

  print("Model Sparsity Summary ({})".format(pruned_model.name))
  print("--")
  for layer in pruned_model.layers:
    if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
      prunable_weights = layer.layer.get_prunable_weights()
      if prunable_weights:
        print("{}: {}".format(
            layer.name, ", ".join([
                "({}, {})".format(weight.name,
                                  str(_get_sparsity(K.get_value(weight))))
                for weight in prunable_weights
            ])))
  print("\n")
          