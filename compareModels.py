import os
from optparse import OptionParser
import pandas as pd
import numpy as np
from sklearn import metrics
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


print("Importing helper libraries")

import h5py
#from sklearn.model_selection import KFold,StratifiedShuffleSplit #Switch to tf.data
import matplotlib.pyplot as plt
from scipy.io import loadmat
from qkeras import quantized_bits
print("Importing private libraries")
import models
from utils import getDatasets,getKfoldDataset, toJSON, parse_config, trainingDiagnostics, performanceSummary,preprocess,print_model_sparsity
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import prune
# from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
# from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
# from tensorflow_model_optimization.python.core.sparsity.keras import prune_low_magnitude
# ConstantSparsity = pruning_schedule.ConstantSparsity



OPTIMIZER   = Adam(lr=0.01, decay=0.000025)
LOSS        = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
BUFFER_SIZE = 1024 
NCLASSES    = 10
  
import matplotlib.pyplot as plt
if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option('-f','--folders',action='store',type='string',dest='folders',default='float_cnn_v1/', help='in folders')
  parser.add_option('-m','--models' ,action='store',type='string',dest='models' ,default='full', help='model')
  parser.add_option('-s','--svhn',action='store_true', dest='svhn', default=True, help='Use SVHN')
  parser.add_option('--mnist',action='store_true', dest='mnist', default=False, help='Use MNIST')
  parser.add_option('-p','--predict',action='store', type='int', dest='predict', default=1, help='Which MNIST number to predict')
  parser.add_option('-w','--doWeights',action='store_true', dest='doWeights', default=False, help='Plot weights')
  parser.add_option('-P','--doProfile',action='store_true', dest='doProfile', default=False, help='Do profile')
  (options,args) = parser.parse_args()
  
  
  X_train, X_test, Y_train, Y_test  = getDatasets(nclasses=10,doMnist=options.mnist,doSvhn=options.svhn,greyScale=False,ext=False)
  folders = [str(f) for f in options.folders.split(',')]
  models  = [str(f) for f in options.models.split(',')]
  
  df = pd.DataFrame()
  
  fpr = {}
  tpr = {}
  auc1 = {}
  scores = []
  labels = []
  for f,m in zip(folders,models):
    scores_ = list()
    for root, dirs, files in os.walk(f, topdown=False):
      for name in dirs:
        if name.find(m)!=-1:
          fullname = (os.path.join(root, name))
          model_name = fullname.split('/')[-1]
          # model_file = f+'/model.json'
        #   with open(model_file) as json_file:
        #       json_config = json_file.read()
        #
          model = tf.keras.models.load_model(fullname+"/saved_model.h5",custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude})
          # model = model_from_json(json_config, custom_objects={
     #                         'PruneLowMagnitude': prune.prune_low_magnitude(),
     #                         'QDense': QDense,
     #                         'QConv2D': QConv2D,
     #                         'QActivation': QActivation,
     #                         'QBatchNormalization': QBatchNormalization})
     #      model.load_weights(f+'/bestWeights.h5')
          #loop over each layer and get weights and biases'
          # plt.figure()
 #          if options.doProfile:
 #            numerical(keras_model=model, X=X_test)
 #          plt.savefig(f+'profile.png')
          if options.doWeights:
              allWeightsByLayer = {}
              for layer in model.layers:
                  print ("----")
                  print (layer._name)
                  if len(layer.get_weights())<1: continue
                  weights = layer.get_weights()[0]
                  weightsByLayer = []
                  for w in weights:
                      weightsByLayer.append(w)
                  if len(weightsByLayer)>0:
                      allWeightsByLayer[layer._name] = np.array(weightsByLayer)
              labelsW = []
              histosW = []

              for key in reversed(sorted(allWeightsByLayer.keys())):
                  labelsW.append(key)
                  histosW.append(allWeightsByLayer[key])

              plt.figure()
              bins = np.linspace(-1.5, 1.5, 50)
              plt.hist(histosW,bins,histtype='step',stacked=False,label=labelsW)
              plt.legend(prop={'size':10}, frameon=False)
              axis = plt.gca()
              ymin, ymax = axis.get_ylim()
              plt.ylabel('Number of Weights')
              plt.xlabel('Weights')
              plt.savefig(f+'%s_weights.png'%m)

          # model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=["accuracy"])


          score        = model.evaluate(X_test, Y_test, verbose=0)
          predict_test = model.predict (X_test)

          print("For model %s" %model_name)
          print ('Keras test score:', score[0])
          print ('Keras test accuracy:', score[1])
    
          df[model_name] = Y_test
          df[model_name + '_pred'] = np.argmax(predict_test,axis=1)
          fpr[model_name], tpr[model_name], threshold = metrics.roc_curve( df[model_name],df[model_name+'_pred'],pos_label=options.predict )

          auc1[model_name] = metrics.auc(fpr[model_name], tpr[model_name])
          # score_file = np.load('float_cnn/scores.npz')
  #         print(score_file.files)
  #         scores_ =score_file['arr_0']
  #         scores.append(scores_)
  #         label = ('$<m>=%.1f$ $\sigma$=%.1f (k=%i)' % (np.mean(scores_)*100, np.std(scores_)*100, len(scores_)))
  #         labels.append(label)
  for model_name in auc1:
      plt.plot(tpr[model_name],fpr[model_name],label='%s, AUC = %.4f'%(model_name.replace("model","").replace("_"," "),auc1[model_name]))
  plt.ylabel("False positive rate")
  plt.xlabel("True positive rate")
  plt.ylim(0.001,1.01)
  plt.xlim(0.9,1.0)
  plt.yscale('log')
  plt.grid(True)
  plt.legend(loc='upper left')
  plt.figtext(0.24, 0.90,'hls4ml',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
  plt.figtext(0.9, 0.9,"y=%i"%options.predict, wrap=True, horizontalalignment='right', fontsize=12)
  plt.savefig('ROC_compare_zy%i.png'%(options.predict))
  performanceSummary(scores,labels, outdir='./',outname='/performance_summary.png')

