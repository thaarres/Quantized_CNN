
import sys, os
import pandas as pd
import numpy as np
from utils import trainingDiagnostics,performanceSummary
from optparse import OptionParser

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option('-o','--outdir',action='store',type='string',dest='outdir',default='plots/', help='output folder')
  parser.add_option('-m','--models' ,action='store',type='string',dest='models' ,default='full', help='model')
  parser.add_option('-n','--names' ,action='store',type='string',dest='names' ,default='float', help='model name')
  parser.add_option('-k','--kfolds',action='store', type='int', dest='kfolds', default=10, help='Number of folds')
  (options,args) = parser.parse_args()
  
  print(" Run with:")
  print('python3 trainingDiagnostics.py -m "float_cnn/full;float_cnn/layerwise_pruning;float_cnn/full_pruning_" --names "Float;Pruned dense;Pruned all"')
  models = [str(f) for f in options.models.split(';')]
  names  = [str(f) for f in options.names.split(';')]
  # indir = options.indir
  nfolds = options.kfolds
  outdir = options.outdir
  folds  = [i for i in range(int(nfolds))]
  
  if not os.path.exists(outdir): os.system('mkdir '+outdir)
  
  val_accuracies_all  = []
  for model,name in zip(models,names):
    histories      = []
    val_accuracies = []
    
    for f in folds:
      history    = pd.read_csv("%s_%i"%(model,f)+"/history_dict.csv") 
      score_file = np.load("%s_%i"%(model,f)+"/scores.npz")
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
    trainingDiagnostics(histories,outdir,'/%s_learning_curve.pdf'%name)
    print("Accuracy mean and spread")
    print (val_accuracies)
    # performanceSummary([val_accuracies],[name],outdir,outname='/%s_performance_summary.pdf'%name)
  
  print(len(val_accuracies_all))
  print(len(names)             )
  performanceSummary(val_accuracies_all,names,outdir,outname='/allModels_performance_summary.pdf')