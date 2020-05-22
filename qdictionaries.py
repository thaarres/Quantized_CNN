allQDictionaries = {}

q_dict_dense_binary={
          'dense_1': {'activation': 'quantized_relu(32,16)',
          'bias_quantizer': 'binary()',
          'kernel_quantizer': 'binary()',
          }}

q_dict_mix = {
    "conv_0": {
        "kernel_quantizer": "binary()",
        "bias_quantizer": "quantized_bits(4,0,1)"
    },
    "conv_1": {
        "kernel_quantizer": "ternary()",
        "bias_quantizer": "quantized_bits(4,0,1)"
    },
    "conv_act_1": "quantized_relu(32,16)",
    "QActivation": {
        "relu": "quantized_relu(32,16)"
    },
    "QConv2D": {
        "kernel_quantizer": "quantized_bits(4,0,1)",
        "bias_quantizer": "quantized_bits(4,0,1)"
    },
    "QDense": {
        "kernel_quantizer": "quantized_bits(3,0,1)",
        "bias_quantizer": "quantized_bits(3,0,1)"
    }}
 
q_dict_binary = {
    "conv_0": {
        "kernel_quantizer": "binary()",
        "bias_quantizer": "binary()"
    },
    "conv_1": {
        "kernel_quantizer": "binary()",
        "bias_quantizer": "binary()"
    },
    "conv_act_1": "quantized_relu(32,16)",
    "QActivation": {
        "relu": "quantized_relu(32,16)"
    },
    "QConv2D": {
        "kernel_quantizer": "binary()",
        "bias_quantizer": "binary()"
    },
    "QDense": {
        "kernel_quantizer": "binary()",
        "bias_quantizer": "binary()"
    }}

q_dict_ternary = {
    "conv_0": {
        "kernel_quantizer": "ternary()",
        "bias_quantizer": "ternary()"
    },
    "conv_1": {
        "kernel_quantizer": "ternary()",
        "bias_quantizer": "ternary()"
    },
    "conv_act_1": "quantized_relu(32,16)",
    "QActivation": {
        "relu": "quantized_relu(32,16)"
    },
    "QConv2D": {
        "kernel_quantizer": "ternary()",
        "bias_quantizer": "ternary()"
    },
    "QDense": {
        "kernel_quantizer": "ternary()",
        "bias_quantizer": "ternary()"
    }}  

q_dict_4bit = {
    "conv_0": {
        "kernel_quantizer": "quantized_bits(4,0,1)",
        "bias_quantizer": "quantized_bits(4,0,1)"
    },
    "conv_1": {
        "kernel_quantizer": "quantized_bits(4,0,1)",
        "bias_quantizer": "quantized_bits(4,0,1)"
    },
    "conv_act_1": "quantized_relu(32,16)",
    "QActivation": {
        "relu": "quantized_relu(32,16)"
    },
    "QConv2D": {
        "kernel_quantizer": "quantized_bits(4,0,1)",
        "bias_quantizer": "quantized_bits(4,0,1)"
    },
    "QDense": {
        "kernel_quantizer": "quantized_bits(4,0,1)",
        "bias_quantizer": "quantized_bits(4,0,1)"
    }} 
q_dict_4bit_binary = {
    "conv_0": {
        "kernel_quantizer": "quantized_bits(4,0,1)",
        "bias_quantizer": "quantized_bits(4,0,1)"
    },
    "conv_1": {
        "kernel_quantizer": "quantized_bits(4,0,1)",
        "bias_quantizer": "quantized_bits(4,0,1)"
    },
    "conv_act_1": "quantized_bits(4,0,1)",
    "QActivation": {
        "relu": "quantized_relu(4,0)"
    },
    "QConv2D": {
        "kernel_quantizer": "quantized_bits(4,0,1)",
        "bias_quantizer": "quantized_bits(4,0,1)"
    },
    "dense_1": {"activation": "quantized_relu(32,16)",
        "kernel_quantizer": "binary()",
        "bias_quantizer": "binary()"
    }}
                        
allQDictionaries['dense_binary']   = q_dict_dense_binary
allQDictionaries['mix']            = q_dict_mix
allQDictionaries['binary']         = q_dict_binary
allQDictionaries['ternary']        = q_dict_ternary 
allQDictionaries['4bit']           = q_dict_4bit  
allQDictionaries['4bitBinary']     = q_dict_4bit_binary       