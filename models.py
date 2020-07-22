from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, Input, Conv2D, MaxPooling2D, Flatten,ZeroPadding2D, AveragePooling2D, GlobalMaxPooling2D, SeparableConv2D, DepthwiseConv2D

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

from tensorflow_model_optimization.sparsity import keras as sparsity
  
def float_cnn(name_, Inputs,nclasses,filters,kernel,strides, pooling, dropout, activation, pruning_params = {}):
  length = len(filters)
  if any(len(lst) != length for lst in [filters, kernel, strides,pooling,dropout]):
    sys.exit("One value for stride and kernel must be added for each filter! Exiting") 
  x = x_in = Inputs
  x = BatchNormalization()(x)
  x = ZeroPadding2D( padding=(1, 1), data_format="channels_last") (x)
  for i,(f,k,s,p,d) in enumerate(zip(filters,kernel,strides,pooling,dropout)):
    print (("Adding layer with {} filters, kernel_size=({},{}), strides=({},{})").format(f,k,k,s,s))
    x = DepthwiseConv2D(int(f),
               name='conv_%i'%i)(x) 
    if float(p) != 0:
      if float(p)==1:
        x = AveragePooling2D()(x)        
      if float(p)==2:
          x = MaxPooling2D()(x)
          
    x = BatchNormalization()(x)
    x = Activation(activation,name='conv_act_%i'%i)(x)
  # x = Flatten()(x) #x = tf.keras.layers.GlobalMaxPooling2D()(x)
#   x = Dense(128,kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001),name='dense_1', use_bias=False)(x)
#   x = Dropout(0.25) (x)
#   x = BatchNormalization()(x)
#   x = Activation(activation,name='dense_act')(x)
  x =  GlobalMaxPooling2D()(x)
  x_out = Dense(nclasses, activation='softmax',name='output')(x)
  model = Model(inputs=[x_in], outputs=[x_out], name=name_)
  return model

def qkeras_cnn(name_, Inputs,nclasses,filters,kernel,strides, pooling, dropout, activation, pruning_params = {},qb=quantized_bits(6,0,alpha=1)):
  length = len(filters)
  if any(len(lst) != length for lst in [filters, kernel, strides,pooling,dropout]):
    sys.exit("One value for stride and kernel must be added for each filter! Exiting") 
  x = x_in = Inputs
  x = BatchNormalization()(x)
  x = ZeroPadding2D( padding=(1, 1), data_format="channels_last") (x)
  for i,(f,k,s,p,d) in enumerate(zip(filters,kernel,strides,pooling,dropout)):
    print (("Adding layer with {} filters, kernel_size=({},{}), strides=({},{})").format(f,k,k,s,s))
    x = QConv2D(int(f), kernel_size=(int(k), int(k)), strides=(int(s),int(s)),
                kernel_quantizer=qb, bias_quantizer=qb,
                kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), use_bias=False, name='conv_%i'%i)(x) 
    if float(p) != 0:
      x = MaxPooling2D(pool_size = (int(p),int(p)) )(x)
    x = BatchNormalization()(x)
    x = Activation(activation,name='conv_act_%i'%i)(x)
  x = Flatten()(x)
  x = QDense(128,kernel_quantizer=qb, bias_quantizer=qb,kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001),name='dense_1', use_bias=False)(x)
  x = Dropout(0.25) (x)
  x = BatchNormalization()(x)
  x = Activation(activation,name='dense_act')(x)
  x_out = Dense(nclasses, activation='softmax',name='output')(x)
  model = Model(inputs=[x_in], outputs=[x_out], name=name_)
  return model
    

def float_cnn_densePrune(name_, Inputs,nclasses,filters,kernel,strides, pooling, dropout, activation, pruning_params = {}):
  length = len(filters)
  if any(len(lst) != length for lst in [filters, kernel, strides,pooling,dropout]):
    sys.exit("One value for stride and kernel must be added for each filter! Exiting") 
  x = x_in = Inputs
  x = BatchNormalization()(x)
  x = ZeroPadding2D( padding=(1, 1), data_format="channels_last") (x)
  for i,(f,k,s,p,d) in enumerate(zip(filters,kernel,strides,pooling,dropout)):
    print (("Adding layer with {} filters, kernel_size=({},{}), strides=({},{})").format(f,k,k,s,s))
    x = Conv2D(int(f), kernel_size=(int(k), int(k)), strides=(int(s),int(s)), kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), use_bias=False,
               name='conv_%i'%i)(x) 
    if float(p) != 0:
      x = MaxPooling2D(pool_size = (int(p),int(p)) )(x)
    x = BatchNormalization()(x)
    x = Activation(activation,name='conv_act_%i'%i)(x)
  x = Flatten()(x)
  x = prune.prune_low_magnitude( (Dense(128,kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), use_bias=False,name='dense_1')),**pruning_params) (x)
  x = Dropout(0.25) (x)
  x = BatchNormalization()(x)
  x = Activation(activation,name='dense_act')(x)
  x_out = Dense(nclasses, activation='softmax',name='output')(x)
  model = Model(inputs=[x_in], outputs=[x_out], name=name_)
  return model
  
  
def float_cnn_allPrune(name_, Inputs,nclasses,filters,kernel,strides, pooling, dropout, activation, pruning_params = {}):
  print ("Building model: float_cnn")
  length = len(filters)
  if any(len(lst) != length for lst in [filters, kernel, strides,pooling,dropout]):
    sys.exit("One value for stride and kernel must be added for each filter! Exiting") 
  x = x_in = Inputs
  x = BatchNormalization()(x)
  x = ZeroPadding2D( padding=(1, 1), data_format="channels_last") (x)
  for i,(f,k,s,p,d) in enumerate(zip(filters,kernel,strides,pooling,dropout)):
    print (("Adding layer with {} filters, kernel_size=({},{}), strides=({},{})").format(f,k,k,s,s))
    x = prune.prune_low_magnitude( (Conv2D(int(f), kernel_size=(int(k), int(k)), strides=(int(s),int(s)), use_bias=False, name='conv_%i'%i)),**pruning_params) (x)
    if float(p) != 0:
      x = MaxPooling2D(pool_size = (int(p),int(p)) )(x)
    x = BatchNormalization()(x)
    x = Activation(activation,name='conv_act_%i'%i)(x)
  x = Flatten()(x)
  x = prune.prune_low_magnitude( (Dense(128,kernel_initializer='lecun_uniform', use_bias=False,name='dense_1')) ,**pruning_params) (x)
  x = BatchNormalization()(x)
  x = Activation(activation,name='dense_act')(x)
  x_out = Dense(nclasses, activation='softmax',name='output')(x)
  model = Model(inputs=[x_in], outputs=[x_out], name=name_)
  return model
  
def float_cnn_1L_Prune(name_, Inputs,nclasses,filters,kernel,strides, pooling, dropout, activation, pruning_params = {}):
  print ("Building model: float_cnn")
  length = len(filters)
  if any(len(lst) != length for lst in [filters, kernel, strides,pooling,dropout]):
    sys.exit("One value for stride and kernel must be added for each filter! Exiting") 
  x = x_in = Inputs
  x = BatchNormalization()(x)
  x = ZeroPadding2D( padding=(1, 1), data_format="channels_last") (x)
  for i,(f,k,s,p,d) in enumerate(zip(filters,kernel,strides,pooling,dropout)):
    print (("Adding layer with {} filters, kernel_size=({},{}), strides=({},{})").format(f,k,k,s,s))
    if i == 1:
      x = prune.prune_low_magnitude( (Conv2D(int(f), kernel_size=(int(k), int(k)), strides=(int(s),int(s)), use_bias=False, name='conv_%i'%i)),**pruning_params) (x)
    else:
      x = Conv2D(int(f), kernel_size=(int(k), int(k)), strides=(int(s),int(s)), kernel_initializer='lecun_uniform', use_bias=False, name='conv_%i'%i)(x)
    if float(p) != 0:
      x = MaxPooling2D(pool_size = (int(p),int(p)) )(x)
    x = BatchNormalization()(x)
    x = Activation(activation,name='conv_act_%i'%i)(x)
  x = Flatten()(x)
  x = Dense(128,kernel_initializer='lecun_uniform', use_bias=False,name='dense_1')(x)
  x = BatchNormalization()(x)
  x = Activation(activation,name='dense_act')(x)
  x_out = Dense(nclasses, activation='softmax',name='output')(x)
  model = Model(inputs=[x_in], outputs=[x_out], name=name_)
  return model
