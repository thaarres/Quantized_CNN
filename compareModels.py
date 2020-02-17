from train import getDatasets,performanceSummary
from qkeras import QDense
from qkeras import QConv2D
from qkeras import QActivation
from qkeras import QBatchNormalization

from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adadelta,Nadam

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option('-f','--folders'   ,action='store',type='string',dest='folders'   ,default='train_float_cnn', help='in folders')
  parser.add_option('-s','--svhn',action='store_true', dest='svhn', default=False, help='Use SVHN')
  parser.add_option('--mnist',action='store_true', dest='mnist', default=False, help='Use MNIST')
  parser.add_option('-p','--predict',action='store', type='int', dest='predict', default=1, help='Which MNIST number to predict')
  parser.add_option('-w','--doWeights',action='store_true', dest='doWeights', default=False, help='Plot weights')
  parser.add_option('-P','--doProfile',action='store_true', dest='doProfile', default=False, help='Do profile')
  (options,args) = parser.parse_args()
  
  
  X_train, X_test, Y_train, Y_test  = getDatasets(nclasses=10,mnist=options.mnist,svhn=options.svhn)
  
  inputs = [str(f) for f in options.folders.split(',')]
  
  df = pd.DataFrame()
  
  fpr = {}
  tpr = {}
  auc1 = {}
  
  for f in folders:
    
      model_file = f+'/model.json'
      with open(model_file) as json_file:
          json_config = json_file.read()
     
      model = model_from_json(json_config, custom_objects={
                         'QDense': QDense,
                         'QConv2D': QConv2D,
                         'QActivation': QActivation,
                         'QBatchNormalization': QBatchNormalization})
      model.load_weights(outdir+'/bestWeights.h5')
      # loop over each layer and get weights and biases'
      plt.figure()
      if options.doProfile:
        numerical(keras_model=model, X=X_test)
      plt.savefig('profile_%s.png'%(model_name))
      if options.doWeights:
          allWeightsByLayer = {}
          for layer in model.layers:
              print ("----")
              print (layer.name)
              if len(layer.get_weights())<1: continue
              weights = layer.get_weights()[0]
              weightsByLayer = []
              for w in weights:                
                  weightsByLayer.append(w)
              if len(weightsByLayer)>0:
                  allWeightsByLayer[layer.name] = np.array(weightsByLayer)    
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
          plt.savefig('weights_%s.png'%(model_name))
          
      model.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=0.0001, decay=0.000025), metrics=['accuracy'])
      
      
      score        = model.evaluate(X_test, Y_test, verbose=0)
      predict_test = model.predict (X_test)
      
      print("For model %s" %model_name)
      print ('Keras test score:', score[0])
      print ('Keras test accuracy:', score[1])
      df[model_name] = Y_test[:,options.predict]
      df[model_name + '_pred'] = predict_test[:,options.predict]
        
      fpr[model_name], tpr[model_name], threshold = roc_curve(df[model_name],df[model_name+'_pred'])
        
      auc1[model_name] = auc(fpr[model_name], tpr[model_name])
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
      
      