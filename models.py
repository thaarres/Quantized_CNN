from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from qkeras import QActivation
from qkeras import QDense, QConv2D
from qkeras import quantized_bits
from qkeras import QBatchNormalization
import sys

def float_cnn(Inputs,nclasses,filters,kernel,strides, pooling, dropout, activation):
  length = len(filters)
  if any(len(lst) != length for lst in [filters, kernel, strides,pooling,dropout]):
    sys.exit("One value for stride and kernel must be added for each filter! Exiting") 
   
  x = x_in = Inputs
  for i,(f,k,s,p,d) in enumerate(zip(filters,kernel,strides,pooling,dropout)):
    print (("Adding layer with {} filters, kernel_size=({},{}), strides=({},{})").format(f,k,k,s,s))
    x = Conv2D(int(f), kernel_size=(int(k), int(k)), strides=(int(s),int(s)),
               name='conv_%i'%i)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPooling2D(pool_size = int(p))(x)
    # x = Dropout(float(d))(x)
  x = Flatten()(x)
  x = Dense(nclasses)(x)
  x = Activation("softmax")(x)
  model = Model(inputs=[x_in], outputs=[x])
  return model

def quantized_cnn(Inputs,nclasses,filters,kernel,strides, pooling, dropout, activation="quantized_relu(32,16)",quantizer_cnn=quantized_bits(1),quantizer_dense=quantized_bits(1)):
   
   length = len(filters)
   if any(len(lst) != length for lst in [filters, kernel, strides, pooling, dropout]):
     sys.exit("One value for stride and kernel must be added for each filter! Exiting") 
    
   x = x_in = Inputs
   for i,(f,k,s,p,d) in enumerate(zip(filters,kernel,strides,pooling,dropout)):
     print (("Adding layer with {} filters, kernel_size=({},{}), strides=({},{})").format(f,k,k,s,s))
     x = QConv2D(int(f), kernel_size=(int(k), int(k)), strides=(int(s),int(s)),
                kernel_quantizer = quantizer_cnn,
                bias_quantizer   = quantizer_cnn,
                name='conv_%i'%i)(x)
     x = BatchNormalization()(x)
     x = QActivation(activation)(x)
     x = MaxPooling2D(pool_size = int(p))(x)
     # x = Dropout(float(d))(x)
   x = Flatten()(x)
   x = QDense(nclasses,
              kernel_quantizer = quantizer_dense,
              bias_quantizer   = quantizer_dense)(x)
   x = Activation("softmax")(x)
   model = Model(inputs=[x_in], outputs=[x])
   return model
