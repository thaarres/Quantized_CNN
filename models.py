from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from qkeras import QActivation
from qkeras import QDense, QConv2D
from qkeras import quantized_bits
from qkeras import QBatchNormalization
import sys

def float_cnn(Inputs,nclasses,filters,kernel,strides,activation):
  x = Conv2D(int(filters[0]),
             kernel_size=(int(kernel[0]),int(kernel[0])),
             strides=(int(strides[0]),int(strides[0])),
             name='conv_0')(Inputs)
  x = Activation(activation)(x)
  x = BatchNormalization()(x)
  # x = MaxPooling2D(pool_size = 2)(x)
  # x = Dropout(0.04)(x)
  for i,(f,k,s) in enumerate(zip(filters,kernel,strides)):
    print (("Adding layer with {} filters, kernel_size=({},{}), strides=({},{})").format(f,k,k,s,s))
    x = Conv2D(int(f),
               kernel_size=(int(k), int(k)),
               strides=(int(s),int(s)),
               name='conv_%i'%(i+1))(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    # x = MaxPooling2D( pool_size = 2)(x)
    # x = Dropout(0.04)(x)
  x = Flatten()(x)
  x = Dense(28, activation='selu',name='dense0')(x)
  x = Dense(28, activation='selu',name='dense1')(x)
  x = Dropout(0.04,name='drop')(x)
  predictions = Dense(nclasses, activation='softmax',name='output')(x)
  model = Model(inputs=Inputs, outputs=predictions)
  return model 

def quantized_cnn(Inputs,nclasses,filters,kernel,strides,activation="quantized_relu(32,16)",quantizer_dense=quantized_bits(1),quantizer_cnn=quantized_bits(1)):
  x = QConv2D(int(filters[0]),
             kernel_size=(int(kernel[0]),int(kernel[0])),
             strides=(int(strides[0]),int(strides[0])),
             kernel_quantizer=quantizer_cnn,
             bias_quantizer  =quantizer_cnn,
             name='conv_0')(Inputs)
  x = QActivation(activation)(x)
  x = BatchNormalization()(x)
  # x = MaxPooling2D(pool_size = 2)(x)
  x = Dropout(0.04)(x)
  for i,(f,k,s) in enumerate(zip(filters,kernel,strides)):
    print (("Adding layer with {} filters, kernel_size=({},{}), strides=({},{})").format(f,k,k,s,s))
    x = QConv2D(int(f),
               kernel_size=(int(k), int(k)),
               strides=(int(s),int(s)),
               kernel_quantizer=quantizer_cnn,
               bias_quantizer  =quantizer_cnn,
               name='conv_%i'%(i+1))(x)
    x = QActivation(activation)(x)
    x = BatchNormalization()(x)
    # x = MaxPooling2D( pool_size = 2)(x)
    x = Dropout(0.04)(x)
  x = Flatten()(x)
  x = QDense( 28,
              kernel_quantizer=quantizer_dense,
              bias_quantizer  =quantizer_dense, 
              use_bias=False,
              name='dense0')(x)
  x = QActivation(activation,name='act_0')(x)
  x = QDense( 28,
              kernel_quantizer=quantizer_dense,
              bias_quantizer  =quantizer_dense, 
              use_bias=False,
              name='dense1')(x)
  x = QActivation(activation,name='act_last')(x)
  predictions = Dense(nclasses, activation='softmax',name='output')(x)
  model = Model(inputs=Inputs, outputs=predictions)
  return model 

# def quantized_cnn(Inputs,nclasses,filters,kernel,strides,activation="quantized_relu(32,16)",quantizer_dense=quantized_bits(1),quantizer_cnn=quantized_bits(1)):
#    x = Inputs
#    for f,k,s in zip(filters,kernel,strides):
#      print (("Adding layer with {} filters, kernel_size=({},{}), strides=({},{})").format(f,k,k,s,s))
#      x = QConv2D(int(f),
#                 kernel_size=(int(k), int(k)),
#                 strides=(int(s),int(s)),
#                 kernel_quantizer=quantizer_cnn,
#                 bias_quantizer  =quantizer_cnn,
#                 name='conv_%s'%f)(x)
#      x = QActivation(activation, name="act%s"%f)(x)
#      x = QBatchNormalization()(x)
#    x = Flatten()(x)
#    x = QDense( 128,
#                kernel_quantizer=quantizer_dense,
#                bias_quantizer  =quantizer_dense,
#                use_bias=False,
#                name='dense1')(x)
#    x = QActivation(activation,name='act_last')(x)
#    #x = Dropout(0.5,name='drop')(x)
#    predictions = Dense(nclasses, activation='softmax',name='output')(x)
#    model = Model(inputs=Inputs, outputs=predictions)
#    return model
#