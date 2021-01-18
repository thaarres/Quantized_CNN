
import sys, os
import pandas as pd
import numpy as np
from utils.utils import trainingDiagnostics,performanceSummary, performanceSummary2
from optparse import OptionParser
from utils.utils import preprocess,add_logo
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras import QConv2D, QDense, Clip, QActivation, QInitializer, quantized_bits, ternary, binary
import tensorflow_datasets as tfds
import tensorflow as tf

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option('-o','--outdir',action='store',type='string',dest='outdir',default='plots/', help='output folder')
  parser.add_option('-m','--models' ,action='store',type='string',dest='models' ,default='full', help='model')
  parser.add_option('-p','--prefix' ,action='store',type='string',dest='prefix' ,default='all', help='model')
  parser.add_option('-e','--extra' ,action='store',type='string',dest='extra' ,default='', help='model')
  parser.add_option('-n','--names' ,action='store',type='string',dest='names' ,default='float', help='model name')
  parser.add_option('-k','--kfolds',action='store', type='int', dest='kfolds', default=10, help='Number of folds')
  (options,args) = parser.parse_args()
  
  print(" Run with:")
  print("python3 trainingDiagnostics.py -m 'models/pruned_quant_1bit;models/pruned_quant_2bit;models/pruned_quant_3bit;models/pruned_quant_4bit;models/pruned_quant_6bit;models/pruned_quant_8bit;models/pruned_quant_10bit;models/pruned_quant_12bit;models/pruned_quant_14bit;models/pruned_quant_16bit;models/pruned_bayesian_v2_best_boosted;models/pruned_full' --names 'B;T;3;4;6;8;10;12;14;16;AQP;BP' --prefix 'quantized_pruned' --extra 'Pruned (50% sparsity)'")
  #python3 trainingDiagnostics.py -m 'models/quant_1bit;models/quant_2bit;models/quant_3bit;models/quant_4bit;models/quant_6bit;models/quant_8bit;models/quant_10bit;models/quant_12bit;models/quant_14bit;models/quant_16bit;models/bayesian_v2_best_boosted;models/full' --names 'B;T;3;4;6;8;10;12;14;16;AQ;BF' --prefix 'quantized' --extra ''
  models = [str(f) for f in options.models.split(';')]
  names  = [str(f) for f in options.names.split(';')]
  # indir = options.indir
  nfolds = options.kfolds
  outdir = options.outdir
  
  
  test_data = tfds.load("svhn_cropped", split='test', as_supervised=True)
  test_data   = test_data .map(preprocess).batch(4096)
  if nfolds>0:
    folds  = [i for i in range(int(nfolds))]
  
    if not os.path.exists(outdir): os.system('mkdir '+outdir)
  
    test_accuracies_all  = []
    pruned_test_accuracies_all  = []
    for model_name,name in zip(models,names):
      histories      = []
      test_accuracies = []
      pruned_test_accuracies = []
      
      for f in folds:
        print("Working on fold: ",f)
        print("Loading model in: ","%s_%i"%(model_name,f) )
        # history    = pd.read_csv("%s_%i"%(model_name,f)+"/history_dict.csv")
        # baselinemodel=tf.keras.models.load_model("models/full_%i"%(f)+"/model_best.h5",custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation, 'QInitializer': QInitializer, 'quantized_bits': quantized_bits, 'ternary': ternary, 'binary': binary})
        model  = tf.keras.models.load_model("%s_%i"%(model_name,f)+"/model_best.h5",custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation, 'QInitializer': QInitializer, 'quantized_bits': quantized_bits, 'ternary': ternary, 'binary': binary})
        pmodel = tf.keras.models.load_model("%s_%i"%(model_name.replace("/","/pruned_"),f)+"/model_best.h5",custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation, 'QInitializer': QInitializer, 'quantized_bits': quantized_bits, 'ternary': ternary, 'binary': binary})
        score = model.evaluate(test_data)
        pscore = pmodel.evaluate(test_data)
        # baseline_score = baselinemodel.evaluate(test_data)
        print("Done evaluating model {}".format(model.name))
        print('\n Test loss:', score[0])
        print('\n Test accuracy:', score[1])
        test_loss = score[0]#/baseline_score[0]
        test_accuracy = score[1]#/baseline_score[1]
        # histories.append(history)
        test_accuracies.append(test_accuracy)
        pruned_test_accuracies.append(pscore[1])
      test_accuracies_all.append(test_accuracies)
      pruned_test_accuracies_all.append(pruned_test_accuracies)
      print("Plotting loss and accuracy")
      # trainingDiagnostics(histories,outdir,'/%s_learning_curve.pdf'%name)
      print("Accuracy mean and spread")
      print (test_accuracies)

    # performanceSummary(test_accuracies_all,names,outdir,'/{}_performance_summary.pdf'.format(options.prefix),options.kfolds,options.extra)
    performanceSummary2(test_accuracies_all,pruned_test_accuracies_all,names,outdir,'/{}_performance_summary_COMBO.pdf'.format(options.prefix),options.kfolds,options.extra)
  
  else:
    if not os.path.exists(outdir): os.system('mkdir '+outdir)
  
    test_accuracies_all  = []
    for model_name,name in zip(models,names):
      print("Reading: ", "%s"%(model_name)+"/history_dict.csv")
      # histories      = []
      test_accuracies = []
    
      model = tf.keras.models.load_model(model_name+"/model_best.h5",custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation, 'QInitializer': QInitializer, 'quantized_bits': quantized_bits})
      score = model.evaluate(test_data)
      print("Done evaluating model {}".format(model.name))
      print('\n Test loss:', score[0])
      print('\n Test accuracy:', score[1])
      test_loss = score[0]
      test_accuracy = score[1]
      # histories.append(history)
      test_accuracies.append(test_accuracy)
      test_accuracies_all.append(test_accuracies)
      print("Plotting loss and accuracy")
      # trainingDiagnostics(histories,outdir,'/%s_learning_curve.pdf'%name)
      print("Accuracy mean and spread")
      print (test_accuracies)
      performanceSummary([test_accuracies],[name],outdir,'/%s_performance_summary.pdf'%name,options.kfolds,options.extra)

    performanceSummary(test_accuracies_all,names,outdir,'/{}_performance_summary.pdf'.format(options.prefix),options.kfolds,options.extra)
    
    