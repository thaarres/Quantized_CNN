import os
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
from utils import getDatasets,getKfoldDataset, toJSON, parse_config, trainingDiagnostics, performanceSummary,preprocess,print_model_sparsity,add_logo
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

  
import matplotlib.pyplot as plt
if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option('-f','--folders',action='store',type='string',dest='folders',default='float_cnn/', help='in folders')
  parser.add_option('-m','--models' ,action='store',type='string',dest='models' ,default='full_0', help='model')
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
          model = tf.keras.models.load_model(fullname+"/saved_model.h5",custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude})

          if options.doProfile:
            plt.figure()
            wp, ap = numerical(keras_model=model, X=X_test[:1000])
            # plt.show()
            wp.savefig('%s_profile_weights.png'%m)
            ap.savefig('%s_profile_activations.png'%m)

          if options.doWeights:
            zeroWeights    = 0
            nonzeroWeights = 0
            allWeightsByLayer = {}
            for layer in model.layers:
              if (layer._name).find("batch")!=-1 or len(layer.get_weights())<1:
                continue
              weights = layer.get_weights()[0]
              weightsByLayer = []
              for w in weights:
                weightsByLayer.append(w)
                if w == 0: zeroWeights +=1
                else: nonzeroWeights +=1
              if len(weightsByLayer)>0:
                allWeightsByLayer[layer._name.replace('prune_low_magnitude_','')] = np.array(weightsByLayer)
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
            fig.tight_layout()
            plt.figtext(0.925, 0.94,m.replace('_',' '), wrap=True, horizontalalignment='right')
            add_logo(ax, fig, 0.14, position='upper right')
            bins = np.linspace(-1.5, 1.5, 50)
            colors = sns.color_palette("colorblind", len(histosW))
            ax.hist(histosW,bins,histtype='stepfilled',stacked=True,label=labelsW,color=colors, edgecolor='black')
            ax.legend(prop={'size':10}, frameon=False)
            axis = plt.gca()
            ymin, ymax = axis.get_ylim()
            plt.ylabel('Number of Weights')
            plt.xlabel('Weights')
            plt.savefig('%s_weights.png'%m)

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

