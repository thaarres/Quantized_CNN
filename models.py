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
  x = Inputs
  for f,k,s in zip(filters,kernel,strides):
    print (("Adding layer with {} filters, kernel_size=({},{}), strides=({},{})").format(f,k,k,s,s))
    x = Conv2D(int(f),
               kernel_size=(int(k), int(k)),
               strides=(int(s),int(s)), 
               activation=activation,
               name='conv_%s'%f)(x)
 
  # x = Conv2D(32, kernel_size=(2, 2),strides=(2,2), activation='relu',name='conv0')(Inputs)
 #  x = Conv2D(64, kernel_size=(2, 2),strides=(2,2), activation='relu',name='conv1')(x)
  #x = MaxPooling2D(pool_size=(2, 2),name='mp1')(x)
  #x = Dropout(0.25,name='drop1')(x)
  x = Flatten()(x)
  x = Dense(128, activation='relu',name='dense')(x)
  #x = Dropout(0.5,name='drop')(x)
  predictions = Dense(nclasses, activation='softmax',name='output')(x)
  model = Model(inputs=Inputs, outputs=predictions)
  return model 

def quantized_cnn(Inputs,nclasses,filters,kernel,strides,activation="quantized_relu(32,16)",quantizer_dense=quantized_bits(1),quantizer_cnn=quantized_bits(1)):
   x = Inputs
   for f,k,s in zip(filters,kernel,strides):
     print (("Adding layer with {} filters, kernel_size=({},{}), strides=({},{})").format(f,k,k,s,s))
     x = QConv2D(int(f),
                kernel_size=(int(k), int(k)),
                strides=(int(s),int(s)), 
                data_format="channels_first",
                kernel_quantizer=quantizer_cnn,
                bias_quantizer  =quantizer_cnn,
                name='conv_%s'%f)(x)
     x = QActivation("quantized_relu(4,0)", name="act2_m")(x)
   x = Flatten()(x)
   x = QDense( 128,
               kernel_quantizer=quantizer_dense,
               bias_quantizer  =quantizer_dense, 
               data_format="channels_first",
               use_bias=False,
               name='dense1')(x)
   x = QActivation(activation,name='act3')(x)
   #x = Dropout(0.5,name='drop')(x)
   predictions = Dense(nclasses, activation='softmax',name='output')(x)
   model = Model(inputs=Inputs, outputs=predictions)
   return model
  