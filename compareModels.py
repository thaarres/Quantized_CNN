import os
print("Set TF logging level to minimum (INFO and WARNING messages are not printed)")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #This is buggy after switching to absl, set manually with SET TF_CPP_MIN_LOG_LEVEL=3

from optparse import OptionParser
import pandas as pd
import numpy as np
from sklearn import metrics
print("Importing TensorFlow")
import tensorflow as tf

import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')

print("Importing helper libraries")
import seaborn as sns

import h5py
import matplotlib.pyplot as plt
from hls4ml.model.profiling import numerical
print("Importing private libraries")
import models
from utils import preprocess,add_logo
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from util import profile
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

def doOPS(model):
  print("Counting number of OPS in model")
  model.summary()
  layer_name, layer_flops, inshape, weights = profile(model)
  for name, flop, shape, weight in zip(layer_name, layer_flops, inshape, weights):
      print("layer:", name, shape, " MegaFLOPS:", flop/1e6, " MegaWeights:", weight/1e6)
  totalGFlops = sum(layer_flops)/1e9
  print("Total FLOPS[GFLOPS]:",totalGFlops )
  return totalGFlops
  
def doWeights(model,outdir="plots/"):
  zeroWeights    = 0
  nonzeroWeights = 0
  allWeightsByLayer = {}
  for layer in model.layers:
    layername = layer._name
    if layername.find("prune")!=-1:
      layername = layername.replace('prune_low_magnitude_','').replace('_',' ')
      layername = layername + ' (Pruned)' 
       
    if (layer._name).find("batch")!=-1 or len(layer.get_weights())<1:
      continue
    weights = layer.get_weights()[0]
    weightsByLayer = []
    for w in weights:
      weightsByLayer.append(w)
    if len(weightsByLayer)>0:
      allWeightsByLayer[layername] = np.array(weightsByLayer)
  labelsW = []
  histosW = []
  
  print("Number of zero-weights = {}".format(zeroWeights))
  print("Number of non-zero-weights = {}".format(nonzeroWeights))
  for key in reversed(sorted(allWeightsByLayer.keys())):
    labelsW.append(key)
    histosW.append(allWeightsByLayer[key])

  plt.figure()
  fig,ax = plt.subplots()
  # plt.semilogy()
  plt.legend(loc='upper left',fontsize=15)
  plt.grid(color='0.8', linestyle='dotted')
  plt.figtext(0.125, 0.18,model_name.replace('_',' ').replace('0',''), wrap=True, horizontalalignment='left',verticalalignment='center')
  add_logo(ax, fig, 0.14, position='upper right')
  bins = np.linspace(-1.5, 1.5, 50)
  colors = sns.color_palette("colorblind", len(histosW))
  ax.hist(histosW,bins,histtype='stepfilled',stacked=True,label=labelsW,color=colors, edgecolor='black')
  ax.legend(prop={'size':10}, frameon=False)
  axis = plt.gca()
  ymin, ymax = axis.get_ylim()
  plt.ylabel('Number of Weights')
  plt.xlabel('Weights')
  plt.savefig(outdir+'/%s_weights.pdf'%model.name)
  
def doProfiling(model,X_test,outdir="plots/"):
  plt.figure()
  wp, ap = numerical(keras_model=model, X=X_test[:1000])
  # plt.show()
  wp.savefig(outdir+'/%s_profile_weights.pdf'%m)
  ap.savefig(outdir+'/%s_profile_activations.pdf'%m)
    
def makeRocs(features_val, labels, labels_val, model, outputDir='plots/'):
  
    predict_test = model.predict(features_val)
    df = pd.DataFrame()
    
    fpr = {}
    tpr = {}
    auc1 = {}
    
    plt.figure()  
    fig,ax = plt.subplots()     
    for i, label in enumerate(labels):
        df[label] = labels_val[:,i]
        df[label + '_pred'] = predict_test[:,i]
    
        fpr[label], tpr[label], threshold = metrics.roc_curve(df[label],df[label+'_pred'])
        auc1[label] = metrics.auc(fpr[label], tpr[label])
        plt.plot(tpr[label],fpr[label],label='%s, AUC = %.1f%%'%(label.replace('j_',''),auc1[label]*100.))
    plt.semilogy()
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Background Efficiency")
    plt.ylim(0.0005,1)
    plt.xlim(0.2,1)
    plt.grid(True)
    plt.legend(loc='upper left')
    add_logo(ax, fig, 0.14, position='upper right')
    plt.figtext(0.125, 0.18,model_name.replace('_',' ').replace('0',''), wrap=True, horizontalalignment='left',verticalalignment='center')
    plt.savefig(outputDir+'%s_ROC.pdf'%model.name)
   
if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option('-n','--names' ,action='store',type='string',dest='names' ,default='float', help='model name')
  parser.add_option('-m','--models' ,action='store',type='string',dest='models' ,default='float_cnn/full_0', help='model')
  parser.add_option('-p','--predict',action='store', type='int', dest='predict', default=1, help='Which number to predict')
  parser.add_option('-w','--doWeights',action='store_true', dest='doWeights', default=False, help='Plot weights')
  parser.add_option('-P','--doProfile',action='store_true', dest='doProfile', default=False, help='Do profile')
  parser.add_option('-O','--doOPS',action='store_true', dest='doOPS', default=False, help='Count OPS')
  parser.add_option('-R','--doROC',action='store_true', dest='doROC', default=False, help='Plot ROC curves')
  (options,args) = parser.parse_args()
  
  print(" Run with:")
  print(' python3 compareModels.py -m "float_cnn/full_0;float_cnn/layerwise_pruning_0;float_cnn/full_pruning_0;float_cnn/1L_pruning_0" --names "Unpruned;Pruned dense;Pruned all;Pruned conv 1" -w -R ')

  (img_train, label_train), (img_test, label_test) = tfds.load("svhn_cropped", split=['train', 'test'], batch_size=-1, as_supervised=True,)
  del (img_train, label_train)
  X_test, Y_test = preprocess(img_test, label_test)
  models  = [str(f) for f in options.models.split(';')]
  names   = [str(f) for f in options.names.split(';')]
  df = pd.DataFrame()
  
  for m,model_name in zip(models,names):
    scores_ = list()
    model = tf.keras.models.load_model(m+"/saved_model.h5",custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude})
    
    if model.name.find('prune')!=-1:
      options.doOPS = False
    if options.doOPS:
      totalGFlops = doOPS(model)
    
    if options.doProfile:
      doProfiling(model,X_test)
      
    if options.doWeights:
      doWeights(model)
    if options.doROC:
      labels=['%i'%nr for nr in range (0,10)]
      makeRocs(X_test, labels, Y_test, model)
 