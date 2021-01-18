import sys
# from qDenseCNN import qDenseCNN
import numpy as np
import hls4ml
import tensorflow as tf
from qkeras.utils import _add_supported_quantized_objects
co = {}
_add_supported_quantized_objects(co)

from qkeras import print_qstats
# for automatic quantization
import pprint
from qkeras.autoqkeras import *
from qkeras import *
from qkeras.utils import model_quantize
from qkeras.qtools import run_qtools
from qkeras.qtools import settings as qtools_settings
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras import quantized_bits
from qkeras import QDense, QActivation

from keras_flops import get_flops

def build_baseline(image_size=16, nclasses=5,filters = [8,8,16]):
  inputs = tf.keras.Input((16),name="Input")
  x = QDense(64,
    kernel_quantizer = quantized_bits(4,0,1),
    bias_quantizer = quantized_bits(4,0,1),name="qdense_1")(inputs)
  x = QActivation('quantized_relu(4,2)',name="qact_1")(x)
  x = QDense(32,
    kernel_quantizer = 'ternary',
    bias_quantizer = 'ternary',name="qdense_2")(x)
  x = QActivation('quantized_relu(3,1)',name="qact_2")(x)
  x = QDense(32,
    kernel_quantizer = quantized_bits(2,1,1),
    bias_quantizer = quantized_bits(2,1,1),name="qdense_3")(x)
  x = QActivation('quantized_relu(4,2)',name="qact_3")(x)
  x = QDense(5,
   kernel_quantizer = 'stochastic_binary',
   bias_quantizer = quantized_bits(8,3,1),name="qdense_nclasses")(x)
  predictions = tf.keras.layers.Activation('softmax',name="softmax")(x)
  model = tf.keras.Model(inputs, predictions,name='baseline')
  return model
  
pams = {}
    
pams['nBits_input']  = {'total': 10, 'integer': 3}
pams['nBits_accum']  = {'total': 11, 'integer': 3}
pams['nBits_weight'] = {'total':  5, 'integer': 1}
pams['nBits_encod']  = {'total':  8, 'integer': 1}
pams['channels_first'] = False
pams['shape'] = (4,4,3)
pams['encoded_dim'] = 16
pams['arrange'] = np.arange(48).reshape((3, 16)).transpose().flatten()

# model = build_baseline()
model_name = 'full_0'
model = tf.keras.models.load_model('models/{}/model_best.h5'.format(model_name),custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation})
model.summary()
# print("Q model summary:")
# print_qmodel_summary(model)
# print("Q dictionary:")
# print(get_quantization_dictionary(model))

with open('run_config.json', 'r') as f:
    run_config = json.load(f)
aq = autoqkeras.forgiving_metrics.ForgivingFactorBits(8, 8, 2, config=run_config['quantization_config'])

total_size_params = 0
total_size_acts = 0
for layer in model.layers:
    layer_name = layer.__class__.__name__
    parameters = aq._param_size(layer)
    activations = aq._act_size(layer)
    print("Parameters {}:{}".format(layer.name,parameters))
    print("Activations {}:{}".format(layer.name,activations))
    total_size_params += parameters
    total_size_acts += activations


total_size, p_size, a_size, model_size_dict = aq.compute_model_size(model)

flops = get_flops(model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")



q = run_qtools.QTools(model, process="horowitz", source_quantizers=[quantized_bits(16, 6, 1)], is_inference=False, weights_path=None,keras_quantizer="fp32",keras_accumulator="fp32", for_reference=False)
# q.qtools_stats_print()

# caculate energy of the derived data type map.
energy_dict = q.pe(
    # whether to store parameters in dram, sram, or fixed
    weights_on_memory="sram",
    # store activations in dram or sram
    activations_on_memory="sram",
    # minimum sram size in number of bits. Let's assume a 16MB SRAM.
    min_sram_size=8*16*1024*1024,
    # whether load data from dram to sram (consider sram as a cache
    # for dram. If false, we will assume data will be already in SRAM
    rd_wr_on_io=False)

# get stats of energy distribution in each layer
energy_profile = q.extract_energy_profile(
    qtools_settings.cfg.include_energy, energy_dict)
# extract sum of energy of each layer according to the rule specified in
# qtools_settings.cfg.include_energy
total_energy = q.extract_energy_sum(
    qtools_settings.cfg.include_energy, energy_dict)

pprint.pprint(energy_profile)
print()

print("Total energy: {:.6f} uJ".format(total_energy / 1000000.0))
print("Total bits = {}".format(total_size))
print("Total parameter bits  = {}".format(total_size_params))
print("Total activation bits = {}".format(total_size_acts))

