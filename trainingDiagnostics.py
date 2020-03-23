
import sys
import pandas as pd
import numpy as np
from utils import trainingDiagnostics,performanceSummary
from optparse import OptionParser

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option('-i','--indir',action='store',type='string',dest='indir',default='float_cnn/', help='in folder')
  parser.add_option('-m','--models' ,action='store',type='string',dest='models' ,default='full', help='model')
  parser.add_option('-k','--kfolds',action='store', type='int', dest='kfolds', default=10, help='Number of folds')
  (options,args) = parser.parse_args()
  
  models  = [str(f) for f in options.models.split(',')]
  indir = options.indir
  nfolds = options.kfolds
  outdir = indir
  folds  = [i for i in range(int(nfolds))]
  
  val_accuracies_all  = []
  for model in models:
    histories      = []
    val_accuracies = []
    
    for f in folds:
      history    = pd.read_csv(indir+"/%s_%i"%(model,f)+"/history_dict.csv") 
      score_file = np.load(indir+"/%s_%i"%(model,f)+"/scores.npz")
      score = score_file['arr_0']
      val_loss = score[0]
      val_accuracy = score[1]
      # print("For model in: {}".format(indir+"/%s_%i"%(model,f)))
      # print(history.head())
      # print('Score = {}'.format(score))
      histories.append(history)
      val_accuracies.append(val_accuracy)
    val_accuracies_all.append(val_accuracies)
    print("Plotting loss and accuracy")
    trainingDiagnostics(histories,outdir,'/%s_learning_curve.png'%model)
    print("Accuracy mean and spread")
    print (val_accuracies)
    performanceSummary([val_accuracies],[model],outdir,outname='/%s_performance_summary.png'%model)
    
  performanceSummary(val_accuracies_all,models,outdir,outname='/allModels_performance_summary.png')