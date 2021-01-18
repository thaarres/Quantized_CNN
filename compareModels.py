import os
print("Set TF logging level to minimum (INFO and WARNING messages are not printed)")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #This is buggy after switching to absl, set manually with SET TF_CPP_MIN_LOG_LEVEL=3

from optparse import OptionParser
import pandas as pd
import numpy as np
from sklearn import metrics
print("Importing TensorFlow")
import tensorflow as tf
from scipy import interp

import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')


import h5py
import math
import matplotlib.pyplot as plt
from hls4ml.model.profiling import numerical
print("Importing private libraries")
import models
from utils.utils import preprocess,add_logo
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from utils.floputil import profile
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
import matplotlib as mpl 
mpl.rcParams["yaxis.labellocation"] = 'center'
mpl.rcParams["xaxis.labellocation"] = 'center'

wcols   = ['#a6bddb','#67a9cf','#3690c0','#02818a','#016c59','#014636']
colors  = ['#543005','#8c510a','#bf812d','#dfc27d','#f6e8c3','#c7eae5','#80cdc1','#35978f','#01665e','#003c30']
colors2 = ['#a50026','#d73027','#f46d43','#fdae61','#fee090','#ffffbf','#e0f3f8','#abd9e9','#74add1','#4575b4','#313695']
colors  = ['#a50026','#d73027','#f46d43','#fdae61','#fee090','#e0f3f8','#abd9e9','#74add1','#4575b4','#313695']
colors  = ['#67001f','#b2182b','#d6604d','#f4a582','#fddbc7','#d1e5f0','#92c5de','#4393c3','#2166ac','#053061']
# colors  = ['#7f3b08','#b35806','#e08214','#fdb863','#fee0b6','#d8daeb','#b2abd2','#8073ac','#542788','#2d004b']
# colors  = ['#40004b','#762a83','#9970ab','#c2a5cf','#e7d4e8','#d9f0d3','#a6dba0','#5aae61','#1b7837','#00441b']
# colors  = ['#8e0152','#c51b7d','#de77ae','#f1b6da','#fde0ef','#e6f5d0','#b8e186','#7fbc41','#4d9221','#276419']
# colors  = ['#543005','#8c510a','#bf812d','#dfc27d','#f6e8c3','#c7eae5','#80cdc1','#35978f','#01665e','#003c30']
from qkeras import QConv2D, QDense, Clip, QActivation, QInitializer, quantized_bits
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

  allWeightsByLayer = {}
  for layer in model.layers:
    print(layer._name)
    if (layer._name).find("batch")!=-1 or len(layer.get_weights())<1:
      continue
    layername = layer._name.replace('prune_low_magnitude_','').replace('_',' ').capitalize() 
    if layername.find("prune")!=-1:
      layername = layername + ' (Pruned)' 
    weights=layer.weights[0].numpy().flatten()  
    allWeightsByLayer[layername] = weights
    print('Layer {}: % of zeros = {}'.format(layername,np.sum(weights==0)/np.size(weights)))
  labelsW = []
  histosW = []
  
  for key in reversed(sorted(allWeightsByLayer.keys())):
    labelsW.append(key)
    histosW.append(allWeightsByLayer[key])

  fig = plt.figure()
  ax = fig.add_subplot()
  # plt.semilogy()
  plt.legend(loc='upper left',fontsize=15,frameon=False)
  plt.grid(False)#color='0.8', linestyle='dotted')
  plt.figtext(0.2, 0.38,model_name.replace('_',' ').replace('0',''), wrap=True, horizontalalignment='left',verticalalignment='center')
  add_logo(ax, fig, 0.3, position='upper right')
  bins = np.linspace(-1.5, 1.5, 50)
  ax.hist(histosW,bins,histtype='stepfilled',stacked=True,label=labelsW,color=wcols, edgecolor='black')
  ax.legend(frameon=False,loc='upper left')
  axis = plt.gca()
  ymin, ymax = axis.get_ylim()
  plt.ylabel('Number of Weights')
  plt.xlabel('Weights')
  plt.savefig(outdir+'/%s_weights.pdf'%model.name)
  
def doProfiling(model,X_test,outdir="plots/"):
  plt.clf()
  wp, ap = numerical(keras_model=model, X=X_test[:1000])
  # plt.show()
  wp.savefig(outdir+'/%s_profile_weights.pdf'%model.name)
  ap.savefig(outdir+'/%s_profile_activations.pdf'%model.name)

def getSingleRoc(features_val, labels, labels_val, model):
  predict_test = model.predict(features_val)
  score = model.evaluate(features_val,labels_val)
  # baseline_score = baselinemodel.evaluate(test_data)
  print("Done evaluating model {}".format(model.name))
  print('\n Test loss:', score[0])
  print('\n Test accuracy:', score[1])
  df = pd.DataFrame()
  fpr  = {}
  tpr  = {}
  auc1 = {}
   
  for i, label in enumerate(labels):
      df[label] = labels_val[:,i]
      df[label + '_pred'] = predict_test[:,i]
  
      fpr[label], tpr[label], threshold = metrics.roc_curve(df[label],df[label+'_pred'])
      auc1[label] = metrics.auc(fpr[label], tpr[label])
  
  return df,fpr, tpr, auc1, score[1] 
  
def makeRocs(features_val, labels, labels_val, model, outputDir='plots/'):
  
    predict_test = model.predict(features_val)
    df = pd.DataFrame()
    
    fpr = {}
    tpr = {}
    auc1 = {}
    
    plt.clf()  
    fig,ax = plt.subplots()     
    for i, label in enumerate(labels):
        df[label] = labels_val[:,i]
        df[label + '_pred'] = predict_test[:,i]
    
        fpr[label], tpr[label], threshold = metrics.roc_curve(df[label],df[label+'_pred'])
        auc1[label] = metrics.auc(fpr[label], tpr[label])
        plt.plot(tpr[label],fpr[label],label='%s, AUC = %.2f%%'%(label.replace('j_',''),auc1[label]*100.),color=colors[i])
    plt.semilogy()
    plt.xlabel("TPR")
    plt.ylabel("FPR")
    plt.ylim(0.0005,1)
    plt.xlim(0.2,1)
    plt.grid(True)
    plt.legend(loc='upper left',frameon=False)
    add_logo(ax, fig, 0.14, position='upper right')
    plt.figtext(0.125, 0.18,model_name.replace('_',' ').replace('0',''), wrap=True, horizontalalignment='left',verticalalignment='center')
    print("Saving to: ",outputDir+'/%s_ROC.pdf'%model.name)
    plt.savefig(outputDir+'/%s_ROC.pdf'%model.name)
   
if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option('-n','--names' ,action='store',type='string',dest='names' ,default='Baseline', help='model name')
  parser.add_option('-m','--models' ,action='store',type='string',dest='models' ,default='models/full_0', help='model')
  parser.add_option('-p','--predict',action='store', type='int', dest='predict', default=1, help='Which number to predict')
  parser.add_option('-w','--doWeights',action='store_true', dest='doWeights', default=False, help='Plot weights')
  parser.add_option('-P','--doProfile',action='store_true', dest='doProfile', default=False, help='Do profile')
  parser.add_option('-O','--doOPS',action='store_true', dest='doOPS', default=False, help='Count OPS')
  parser.add_option('-R','--doROC',action='store_true', dest='doROC', default=False, help='Plot ROC curves')
  parser.add_option('--kFold',action='store_true', dest='dokFold', default=False, help='DO kFOld ROCs and accuracy')
  parser.add_option('-o','--outdir',action='store',type='string',dest='outdir',default='plots/', help='output folder')
  (options,args) = parser.parse_args()
  
  print(" Run with:")
  print(' python3 compareModels.py -m "models/pruned_full_0;models/full_0" --names "Baseline Pruned (BP);Baseline Full (BF)" -w --kFold ')
  # python3 compareModels.py -m "models/bayesian_v2_best_boosted_0;models/pruned_bayesian_v2_best_boosted_0" --names "AutoQ (AQ);AutoQ Pruned (AQP)" -w --kFold
  
  
  outdir = options.outdir
  if not os.path.exists(outdir):
    os.system('mkdir '+outdir)
    
  (img_train, label_train), (img_test, label_test) = tfds.load("svhn_cropped", split=['train', 'test'], batch_size=-1, as_supervised=True,)
  del (img_train, label_train)
  X_test, Y_test = preprocess(img_test, label_test)
  models  = [str(f) for f in options.models.split(';')]
  names   = [str(f) for f in options.names.split(';')]
  labels=['%i'%nr for nr in range (0,10)]
  df = pd.DataFrame()
  
  
    
  for m,model_name in zip(models,names):
    print("Loading model: ", model_name)
    scores_ = list()
    model = tf.keras.models.load_model(m+"/model_best.h5",custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation, 'QInitializer': QInitializer, 'quantized_bits': quantized_bits})
    # model.load_weights(m+'/KERAS_check_best_model_weights.h5')
    # model._name = model_name
    
    if options.dokFold:
      folds  = [i for i in range(int(10))]

      fprs = {'0':[],'1':[],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[]}
      tprs = {'0':[],'1':[],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[]}
      aucs = {'0':[],'1':[],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[]}
      accuracies =  []
        
      for fold in folds:
          mname = m.replace("_0","_%i"%fold)
          model_tmp = tf.keras.models.load_model(mname+"/model_best.h5",custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation, 'QInitializer': QInitializer, 'quantized_bits': quantized_bits})

          df,fpr, tpr, auc1, acc = getSingleRoc(X_test, labels, Y_test, model_tmp)
          accuracies.append(acc)
          for i, label in enumerate(labels):
            print('Label = {}. AUC = {}'.format(label,auc1[label]))
            fprs[label].append(fpr [label])
            tprs[label].append(tpr [label])
            aucs[label].append(auc1[label])

      
      fig = plt.figure()
      ax = fig.add_subplot()
      
      for i, label in enumerate(labels):
        
        npoints = 50
        base_fpr = np.exp(np.linspace(math.log(0.0005), 0., npoints))

        First = True
        tpr_array = np.array([])
        auc_array = []
        
        for fold in folds:
          this_fpr         = np.array(fprs[label][fold])
          this_tpr         = np.array(tprs[label][fold])
          tpr_interpolated = interp(base_fpr, this_fpr, this_tpr)
          tpr_interpolated = tpr_interpolated.reshape((1,npoints))
          tpr_array        = np.concatenate([tpr_array, tpr_interpolated], axis=0) if tpr_array.size else tpr_interpolated
          auc_array.append(aucs[label][fold])
          print('aucs[{}][{}]={}'.format(label,fold,aucs[label][fold]))
      
        mean_tpr  = np.mean(tpr_array, axis=0)
        rms_tpr   = np.std(tpr_array, axis=0)
        plus_tpr  = np.minimum(mean_tpr+rms_tpr, np.ones(npoints))
        minus_tpr = np.maximum(mean_tpr-rms_tpr,np.zeros(npoints))
        avg_tpr   = mean_tpr

        auc_array = np.array(auc_array)
        Mean_AUC  = float(np.mean(auc_array, axis=0))
        RMS_AUC   = float(np.std(auc_array, axis=0))
       
        print('label')
        print(label)
        print('Mean_AUC' )
        print(Mean_AUC )
        print('RMS_AUC ')
        print(RMS_AUC)
        print('{}) AUC = {} pm {}'.format(label,Mean_AUC,RMS_AUC))

        plt.plot(base_fpr,avg_tpr,label=r'{} (AUC = {:.4f} $\pm$ {:.4f})'.format(label.replace('j_',''),Mean_AUC,RMS_AUC), linewidth=1.5, color=colors[i])
        plt.fill_between(base_fpr, minus_tpr, plus_tpr, alpha=0.3, color=colors[i])

      plt.semilogx()
      plt.ylabel("True Positive Rate")
      plt.xlabel("False Positive Rate")
      plt.xlim(0.0005,1.)
      plt.ylim(0.2,1.2)
      add_logo(ax, fig, 0.3, position='upper right')
    
        
      median         = np.median(accuracies)
      mean           = np.mean(accuracies)  
      rms            = np.std(accuracies)   
      upper_quartile = np.percentile(accuracies, 75); print(upper_quartile )
      lower_quartile = np.percentile(accuracies, 25); print(lower_quartile )
      plt.figtext(0.2, 0.88,model_name.replace('_',' ').replace('0',''), wrap=True, horizontalalignment='left',verticalalignment='center')
      plt.figtext(0.2, 0.83,r'Accuracy = {:.1f} $\pm$ {:.1f}%'.format(mean*100,rms*100), wrap=True, horizontalalignment='left',verticalalignment='center')
      # plt.figtext(0.17, 0.25,model_name.replace('_',' ').replace('0',''), wrap=True, horizontalalignment='left',verticalalignment='center')
 #      plt.figtext(0.17, 0.20,r'Acc. = {:.1f} $\pm$ {:.1f}%'.format(mean*100,rms*100), wrap=True, horizontalalignment='left',verticalalignment='center')
      plt.legend(loc='lower right',frameon=False)
      plt.draw()
      # plt.figtext(0.127, 0.33,r'Accuracy = {:.2f}%'.format(median*100), wrap=True, horizontalalignment='left',verticalalignment='center')
      # plt.figtext(0.127, 0.28,r'(IQR: {:.2f}-{:.2f})'.format(lower_quartile*100,upper_quartile*100), wrap=True, horizontalalignment='left',verticalalignment='center')
      print("Saving to: ",outdir+'/%s_kfold_ROC.pdf'%model.name)
      plt.savefig(outdir+'/%s_kfold_ROC.pdf'%model.name)
   
      
    if model.name.find('prune')!=-1:
      options.doOPS = False
    if options.doOPS:
      totalGFlops = doOPS(model)

    if options.doProfile:
      doProfiling(model,X_test,outdir)

    if options.doWeights:
      doWeights(model,outdir)
    if options.doROC:
      makeRocs(X_test, labels, Y_test, model,outdir)
 