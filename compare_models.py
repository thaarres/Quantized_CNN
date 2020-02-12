from optparse import OptionParser

import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model,model_from_json
from tensorflow.keras.optimizers import SGD, Adam, RMSprop,Adadelta

from binary_layers import BinaryDense, Clip, DropoutNoScaleForBinary, DropoutNoScale, BinaryConv2D
from binary_ops import binary_tanh as binary_tanh_op

from ternary_layers import TernaryDense, DropoutNoScaleForTernary
from ternary_ops import ternarize

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd

from constraints import ZeroSomeWeights
from sklearn.metrics import accuracy_score

from train import parse_config,get_features
from models import relu1
from qkeras import QDense
from qkeras import QConv2D
from qkeras import QActivation
from qkeras import QBatchNormalization

from hls4ml.model.profiling import numerical

import sys
def binary_tanh(x):
    return binary_tanh_op(x)

def ternary_tanh(x):
    x = K.clip(x, -1, 1)
    return ternarize(x)

if __name__ == "__main__":
    
    parser = OptionParser()
    parser.add_option('-c','--configs',action='store', type='string',dest='configs',default='mnist.yml/', help='list of yml config files')
    parser.add_option('-p','--predict',action='store', type='int', dest='predict', default=1, help='Which MNIST number to predict')
    parser.add_option('-w','--doWeights',action='store_true', dest='doWeights', default=False, help='Plot weights')
    parser.add_option('-P','--doProfile',action='store_true', dest='doProfile', default=False, help='Do profile')
    parser.add_option('-f','--fashionMNIST',action='store_true', dest='fashionMNIST', default=True, help='Use fashion MNIST rather than MNIST')
    (options,args) = parser.parse_args()
    
    nclasses = 10
    if options.fashionMNIST:
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:    
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    #X_test = X_test.reshape(10000, 784)
    X_test = X_test.astype('float32')
    X_test /= 255
    Y_test = to_categorical(y_test, nclasses)
        
    configs = [str(config) for config in options.configs.split(',')]
    
    df = pd.DataFrame()
    
    fpr = {}
    tpr = {}
    auc1 = {}
    
    for cfg in configs:
         
        config = parse_config(cfg)
        if 'cnn' in config['KerasModel']:
            img_rows, img_cols = 28, 28
            if config['DataFormat'] == 'channels_first':
                X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
            else:
                X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        else:
             X_test = X_test.reshape(10000, 784)     
        if config['KerasLoss'] == 'squared_hinge':
            Y_test = to_categorical(y_test, nclasses) * 2 - 1
  
        outdir     = config['OutputDir']
        if options.fashionMNIST:
            outdir = 'fashionMNIST_'+ outdir
        model_name = config['KerasModel']
        model_file = outdir+'/KERAS_model.json'
        with open(model_file) as json_file:
            json_config = json_file.read()
       
        model = model_from_json(json_config, custom_objects={
                           'relu1': relu1,
                           'ZeroSomeWeights' : ZeroSomeWeights,
                           'DropoutNoScale':DropoutNoScale,
                           'DropoutNoScaleForBinary':DropoutNoScaleForBinary,
                           'DropoutNoScaleForTernary':DropoutNoScaleForTernary,
                           'Clip': Clip,
                           'BinaryDense': BinaryDense,
                           'TernaryDense': TernaryDense,
                           'BinaryConv2D': BinaryConv2D,
                           'binary_tanh': binary_tanh,
                           'QDense': QDense,
                           'QConv2D': QConv2D,
                           'QActivation': QActivation,
                           'QBatchNormalization': QBatchNormalization})
        model.load_weights(outdir+'/KERAS_check_best_model_weights.h5')
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
            
        # learning rate schedule
        lr_start = 1e-3
        if not 'cnn' in config['KerasModel']: opt = Adam(lr=lr_start)
        else: opt = Adadelta()
        model.compile(loss=config['KerasLoss'], optimizer=opt, metrics=['acc'])
        
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
    if options.fashionMNIST:
        plt.figtext(0.9, 0.9,"Fashion MNIST y=%i"%options.predict, wrap=True, horizontalalignment='right', fontsize=12)
        plt.savefig('ROC_compare_fashionMNISTy%i.png'%(options.predict))
    else:
        plt.figtext(0.9, 0.9,"MNIST y=%i"%options.predict, wrap=True, horizontalalignment='right', fontsize=12)
        plt.savefig('ROC_compare_MNISTy%i.png'%(options.predict))
        
        
        
        