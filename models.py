from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, Input, Conv2D, MaxPooling2D, Flatten,ZeroPadding2D

from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from qkeras import QActivation
from qkeras import QDense, QConv2D
from qkeras import quantized_bits
from qkeras import QBatchNormalization
import sys

from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule

ConstantSparsity = pruning_schedule.ConstantSparsity


  
def float_cnn(Inputs,nclasses,filters,kernel,strides, pooling, dropout, activation):
  print ("Building model: float_cnn")
  length = len(filters)
  if any(len(lst) != length for lst in [filters, kernel, strides,pooling,dropout]):
    sys.exit("One value for stride and kernel must be added for each filter! Exiting") 

  x = x_in = Inputs
  x = BatchNormalization()(x)
  x = ZeroPadding2D( padding=(1, 1), data_format="channels_last") (x)
  for i,(f,k,s,p,d) in enumerate(zip(filters,kernel,strides,pooling,dropout)):
    print (("Adding layer with {} filters, kernel_size=({},{}), strides=({},{})").format(f,k,k,s,s))
    x = Conv2D(int(f), kernel_size=(int(k), int(k)), strides=(int(s),int(s)),
               name='conv_%i'%i)(x) 
    if float(p) != 0:
      x = MaxPooling2D(pool_size = (int(p),int(p)) )(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    # mlpconvs=[16,16,16]
    # if (i % 2) == 0:
    #   x = Conv2D(mlpconvs[i], kernel_size=(1,1), strides=(1,1),
    #              name='mlpconv_%i'%i)(x)
    #   x = Activation("relu")(x)
    #   x = BatchNormalization()(x)

  x = Flatten()(x)
  # x = Dropout(float(0.3))(x)
  x = Dense(128)(x)
  # x = BatchNormalization()(x)
  x = Activation(activation)(x)
  x = Dense(nclasses)(x)
  # x = Activation("softmax")(x)
  model = Model(inputs=[x_in], outputs=[x])
  return model

def float_cnn_densePrune(Inputs,nclasses,filters,kernel,strides, pooling, dropout, activation,pruning_params = {'pruning_schedule': ConstantSparsity(0.75, begin_step=2000, frequency=100)}):
  print ("Building model: float_cnn_densePrune")
  length = len(filters)
  if any(len(lst) != length for lst in [filters, kernel, strides,pooling,dropout]):
    sys.exit("One value for stride and kernel must be added for each filter! Exiting") 

  x = x_in = Inputs
  x = BatchNormalization()(x)
  x = ZeroPadding2D( padding=(1, 1), data_format="channels_last") (x)
  for i,(f,k,s,p,d) in enumerate(zip(filters,kernel,strides,pooling,dropout)):
    print (("Adding layer with {} filters, kernel_size=({},{}), strides=({},{})").format(f,k,k,s,s))
    x = Conv2D(int(f), kernel_size=(int(k), int(k)), strides=(int(s),int(s)),
               name='conv_%i'%i)(x) 
    if float(p) != 0:
      x = MaxPooling2D(pool_size = (int(p),int(p)) )(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
  
  x = Flatten()(x)
  # x = Dropout(float(0.3))(x)
  x = prune.prune_low_magnitude( (Dense(128)),**pruning_params) (x)
  x = Activation(activation)(x)
  x = Dense(nclasses)(x)
  model = Model(inputs=[x_in], outputs=[x])
  return model

def float_cnn_allPrune(Inputs,nclasses,filters,kernel,strides, pooling, dropout, activation,pruning_params = {'pruning_schedule': ConstantSparsity(0.75, begin_step=2000, frequency=100)}):
  print ("Building model: float_cnn_allPrune")
  length = len(filters)
  if any(len(lst) != length for lst in [filters, kernel, strides,pooling,dropout]):
    sys.exit("One value for stride and kernel must be added for each filter! Exiting") 

  x = x_in = Inputs
  x = BatchNormalization()(x)
  x = ZeroPadding2D( padding=(1, 1), data_format="channels_last") (x)
  
  for i,(f,k,s,p,d) in enumerate(zip(filters,kernel,strides,pooling,dropout)):
    print (("Adding layer with {} filters, kernel_size=({},{}), strides=({},{})").format(f,k,k,s,s))
    x = prune.prune_low_magnitude( (Conv2D(int(f), kernel_size=(int(k), int(k)), strides=(int(s),int(s)), name='conv_%i'%i)),**pruning_params) (x)
    if float(p) != 0:
        x = MaxPooling2D(pool_size = (int(p),int(p)) )(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
  x = Flatten()(x)
  # x = Dropout(float(0.3))(x)
  x = prune.prune_low_magnitude( (Dense(128)),**pruning_params) (x)
  x = Activation(activation)(x)
  x = Dense(nclasses)(x)
  model = Model(inputs=[x_in], outputs=[x])
  return model
    
def quantized_cnn(Inputs,nclasses,filters,kernel,strides, pooling, dropout, activation="quantized_relu(32,16)",quantizer_cnn=quantized_bits(1),quantizer_dense=quantized_bits(1)):
   print ("Building model: quantized_cnn")
   length = len(filters)
   if any(len(lst) != length for lst in [filters, kernel, strides, pooling, dropout]):
     sys.exit("One value for stride and kernel must be added for each filter! Exiting") 
    
   x = x_in = Inputs
   x = BatchNormalization()(x)
   x = ZeroPadding2D( padding=(1, 1), data_format="channels_last") (x)
   
   for i,(f,k,s,p,d) in enumerate(zip(filters,kernel,strides,pooling,dropout)):
     print (("Adding layer with {} filters, kernel_size=({},{}), strides=({},{})").format(f,k,k,s,s))
     x = QConv2D(int(f), kernel_size=(int(k), int(k)), strides=(int(s),int(s)),
                kernel_quantizer = quantizer_cnn,
                bias_quantizer   = quantizer_cnn,
                name='conv_%i'%i)(x)
     if float(p) != 0:
       x = MaxPooling2D(pool_size = (int(p),int(p)) )(x)
     x = QBatchNormalization()(x)
     x = QActivation(activation)(x)
   x = Flatten()(x)
   x = QDense(128,
              kernel_quantizer = quantizer_dense,
              bias_quantizer   = quantizer_dense)(x)
   x = QActivation(activation)(x)
   x = Dense(nclasses)(x)
   model = Model(inputs=[x_in], outputs=[x])
   return model
