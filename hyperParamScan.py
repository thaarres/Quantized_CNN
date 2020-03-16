# Author: T. Aarrestad, CERN
#
#
# ==============================================================================
"""Credits"""

# This code was adapted from
#
# https://github.com/jmduarte/JEDInet-code
#
# and takes advantage of the libraries at
#
# https://github.com/SheffieldML/GPy
# https://github.com/SheffieldML/GPyOpt

from scipy.io import loadmat
import sys
import h5py
import glob
import numpy as np
# keras imports
print("Importing TensorFlow")
import tensorflow
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist
import matplotlib; matplotlib.use('PDF') 
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, Dropout, Flatten
from tensorflow.keras.layers import Concatenate, BatchNormalization, Activation
from tensorflow.keras.layers import MaxPooling2D, MaxPooling3D
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.utils import to_categorical
K.set_image_data_format('channels_last')
# Helper libraries
import GPyOpt
import GPy
import setGPU
#Private libraries
def getDatasets(nclasses,doMnist=False,doSvhn=True,greyScale=False,ext=False):

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
  
    return x_train,x_test,y_train,y_test

####################################################

# myModel class
class myModel():
    def __init__(self, optmizer_index=0, CNN_filters=10, 
                 CNN_filter_size=5, CNN_MaxPool_size=5, CNN_layers=2, CNN_activation_index=0, DNN_neurons=40, 
                 DNN_layers=2, DNN_activation_index=0, dropout=0.2, batch_size=100, epochs=50):  
       
        self.activation = ['relu', 'selu']
        self.optimizer = ['adam', 'nadam','adadelta']
        self.optimizer_index = optmizer_index
        self.CNN_filters = CNN_filters
        self.CNN_filter_size = CNN_filter_size
        self.CNN_MaxPool_size = CNN_MaxPool_size
        self.CNN_layers = CNN_layers
        self.CNN_activation_index =  CNN_activation_index
        self.DNN_neurons = DNN_neurons
        self.DNN_layers = DNN_layers
        self.DNN_activation_index = DNN_activation_index
        self.dropout = dropout
        self.batch_size = batch_size
        # here an epoch is a single file
        self.epochs = epochs
        X_train, X_test, Y_train, Y_test  = getDatasets(nclasses=10,doMnist=False,doSvhn=True)
        self.nclasses    = 10
        self.input_shape = X_train.shape[1:]
        self.__y_test  = Y_test
        self.__y_train = Y_train
        self.__x_test  = X_test
        self.__x_train = X_train
        
        self.__model   = self.build()
    
    #  model
    def build(self):
        inputImage = Input(shape=(self.input_shape))
        x = Conv2D(self.CNN_filters, kernel_size=(self.CNN_filter_size,self.CNN_filter_size), 
                   data_format="channels_last", strides=(1, 1), padding="same", input_shape=self.input_shape,
                   kernel_initializer='lecun_uniform', name='cnn2D_0')(inputImage) #he_uniform
        x = BatchNormalization()(x)
        x = Activation(self.activation[self.CNN_activation_index])(x)
        x = MaxPooling2D( pool_size = (self.CNN_MaxPool_size,self.CNN_MaxPool_size))(x)
        x = Dropout(self.dropout)(x)
        for i in range(1,self.CNN_layers):
            x = Conv2D(self.CNN_filters, kernel_size=(self.CNN_filter_size,self.CNN_filter_size), 
                   data_format="channels_last", strides=(1, 1), padding="same", input_shape=self.input_shape,
                    kernel_initializer='lecun_uniform', name='cnn2D_%i' %i)(x)
            x = BatchNormalization()(x)
            x = Activation(self.activation[self.CNN_activation_index])(x)
            x = MaxPooling2D( pool_size = (self.CNN_MaxPool_size,self.CNN_MaxPool_size))(x)
            x = Dropout(self.dropout)(x)
            
        ####
        x = Flatten()(x)
        #
        for i in range(self.DNN_layers):
            x = Dense(self.DNN_neurons, activation=self.activation[self.DNN_activation_index], 
                      kernel_initializer='lecun_uniform', name='dense_%i' %i)(x)
            x = Dropout(self.dropout)(x)
       
        output = Dense(self.nclasses, activation='softmax', kernel_initializer='lecun_uniform', 
                       name = 'output_softmax')(x)
        ####
        model = Model(inputs=inputImage, outputs=output)
        model.summary()
        model.compile(optimizer=self.optimizer[self.optimizer_index], 
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        return model

    
    # fit model
    def model_fit(self):

        self.__model.fit(self.__x_train, self.__y_train, epochs=self.epochs, 
                                   batch_size= self.batch_size, validation_data=[self.__x_test, self.__y_test],verbose=0, 
                                   callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1),
                                                           ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0), 
                                                           TerminateOnNaN()])
    # evaluate  model
    def model_evaluate(self):
        self.model_fit()
        evaluation = self.__model.evaluate(self.__x_test, self.__y_test, batch_size=self.batch_size, verbose=0)
        return evaluation


####################################################

# Runner function for model
# function to run  class

def run_model(optmizer_index=0, CNN_filters=10, 
              CNN_filter_size=5, CNN_MaxPool_size=2, CNN_layers=2, CNN_activation_index=0, DNN_neurons=40, 
              DNN_layers=2, DNN_activation_index=0, dropout=0.2, batch_size=100, epochs=50):
    
    _model = myModel( optmizer_index, CNN_filters, CNN_MaxPool_size, CNN_filter_size,
                 CNN_layers, CNN_activation_index, DNN_neurons, DNN_layers, DNN_activation_index, 
                 dropout, batch_size, epochs)
    model_evaluation = _model.model_evaluate()
    return model_evaluation

n_epochs = 100

# Bayesian Optimization

# the bounds dict should be in order of continuous type and then discrete type
bounds = [{'name': 'optmizer_index',        'type': 'discrete',   'domain': (0, 1, 2)},
          {'name': 'CNN_filters',           'type': 'discrete',   'domain': (8, 16, 32,64)},
          {'name': 'CNN_filter_size',       'type': 'discrete',   'domain': (3, 3)},
          {'name': 'CNN_MaxPool_size',      'type': 'discrete',   'domain': (1, 2, 3)},
          {'name': 'CNN_layers',            'type': 'discrete',   'domain': (1, 2, 3)},
          {'name': 'CNN_activation_index',  'type': 'discrete',   'domain': (0, 1)},
          {'name': 'DNN_neurons',           'type': 'discrete',   'domain': (28, 64, 128, 256)},
          {'name': 'DNN_layers',            'type': 'discrete',   'domain': (1, 2, 3)},
          {'name': 'DNN_activation_index',  'type': 'discrete',   'domain': (0, 1 )},
          {'name': 'dropout',               'type': 'continuous', 'domain': (0.0, 0.4)},
          {'name': 'batch_size',            'type': 'discrete',   'domain': (32, 50, 200, 500,1000)}]

# function to optimize model
def f(x):
    print "x parameters are"
    print(x)
    evaluation = run_model(optmizer_index       = int(x[:,0]), 
                           CNN_filters          = int(x[:,1]), 
                           CNN_filter_size      = int(x[:,2]),
                           CNN_MaxPool_size     = int(x[:,3]),
                           CNN_layers           = int(x[:,4]), 
                           CNN_activation_index = int(x[:,5]), 
                           DNN_neurons          = int(x[:,6]), 
                           DNN_layers           = int(x[:,7]),
                           DNN_activation_index = int(x[:,8]),
                           dropout              = float(x[:,9]),
                           batch_size           = int(x[:,10]),
                           epochs = n_epochs)
    print("LOSS:\t{0} \t ACCURACY:\t{1}".format(evaluation[0], evaluation[1]))
    print(evaluation)
    return evaluation[0]

if __name__ == "__main__":
  opt_model = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)
  opt_model.run_optimization(max_iter=10000,report_file='bayOpt.txt')
  opt_model.plot_acquisition('bayOpt_acqu.pdf')
  opt_model.plot_convergence('bayOpt_conv.pdf')

  print("DONE")
  print("x:",opt_model.x_opt)
  print("y:",opt_model.fx_opt)

  # print optimized model
  print("""
  Optimized Parameters:
  \t{0}:\t{1}
  \t{2}:\t{3}
  \t{4}:\t{5}
  \t{6}:\t{7}
  \t{8}:\t{9}
  \t{10}:\t{11}
  \t{12}:\t{13}
  \t{14}:\t{15}
  \t{16}:\t{17}
  \t{18}:\t{19}
  \t{20}:\t{21}
  """.format(bounds[0]["name"],opt_model.x_opt[0],
             bounds[1]["name"],opt_model.x_opt[1],
             bounds[2]["name"],opt_model.x_opt[2],
             bounds[3]["name"],opt_model.x_opt[3],
             bounds[4]["name"],opt_model.x_opt[4],
             bounds[5]["name"],opt_model.x_opt[5],
             bounds[6]["name"],opt_model.x_opt[6],
             bounds[7]["name"],opt_model.x_opt[7],
             bounds[8]["name"],opt_model.x_opt[8],
            bounds[9]["name"],opt_model.x_opt[9],
            bounds[10]["name"],opt_model.x_opt[10]
             ))
  print("optimized loss: {0}".format(opt_model.fx_opt))
