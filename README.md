# Quantized CNN training

Training of quantized CNN on MNIST,fashion-MNIST and SVHN datasets. Quantization provided by the Google Keras extension provided at github.com/google/qkeras.
Includes: kFold cross validation training and testing with TF Data, multi-GPU distributed training, hyperparameter optimisation, model ROC/error/accuracy with errors from cross validation

## Dependencies

For training: Python 3.6, TensorFlow version >= 2.1.0, Keras version: 2.2.4-tf, QKeras (https://github.com/google/qkeras).
QKeras is a sub-module of this repositry. To build and install:
```
git clone --recurse-submodules git@github.com:thaarres/Quantized_CNN.git
cd Quantized_CNN/
git submodule update --recursive
cd qkeras/
python3 setup.py build
python3 setup.py install --user
cd ../
```

## Training

To train, flags are set using absl.flags (https://abseil.io/docs/python/guides/flags). One flagfile is provided: svhn_cnn.cfg, use this, pass all parameters from commandline or create a new flag file. Specific architecture (see models.py), number of filters,kernel size, strides, loss function etc. can be set in flagfile or command line

To train use the command:

```
python3 train.py --flagfile=svhn_cnn.cfg #Runs normal full model training
python3 train.py --flagfile=svhn_cnn.cfg --prune=True #Runs pruning for 3 types: prune all, prune dense, prune largest flop consumer
python3 train.py --flagfile=svhn_cnn.cfg --quantize=True #Runs quantization-aware training (runs all maps in qdictionaries.py)
```

Training diagnositics (using k folds)

```
python3 trainingDiagnostics.py -m "float_cnn/full;float_cnn/layerwise_pruning;float_cnn/full_pruning_" --names "Float;Pruned dense;Pruned all" --kfolds 10

```
Plot weights, calculate flops, profile weights and plot ROC
```
python3 compareModels.py -m "float_cnn/full_0;float_cnn/layerwise_pruning_0;float_cnn/full_pruning_0;float_cnn/1L_pruning_0" --names "Unpruned;Pruned dense;Pruned all;Pruned conv 1" -w -R 
```

## Hyperparameter scan (Bayesian Optimisation)

This method depends on the GPy and GPyOpt packages and currently only support Python 2.7 (?). First, install these with

```
pip install gpy --user
pip install GPyOpt --user
```
Then you can do 
```
python2 hyperParamScan.py 
```

The hyperparameters to be optimized are listed in  the dictionary "bounds" and can be configured as desired
