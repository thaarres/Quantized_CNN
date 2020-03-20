# Quantized CNN training

Training of quantized CNN on MNIST,fashion-MNIST and SVHN datasets. Quantization provided by the Google Keras extension provided at github.com/google/qkeras.
Includes: kFold cross validation training and testing with TF Data, multi-GPU distributed training, hyperparameter optimisation, model ROC/error/accuracy with errors from cross validation

## Dependencies

For training: Python 3.6, TensorFlow version >= 2.1.0, Keras version: 2.2.4-tf, QKeras (https://github.com/google/qkeras).
QKeras is a sub-module of this repositry. To build and install:
```
git clone --recurse-submodules -j8 git@github.com:thaarres/Quantized_CNN.git
cd Quantized_CNN/
cd qkeras/
python3 setup.py build
python3 setup.py install --user
cd ../
```

## Training

To train choose one of the provided *.yaml config files or create a new one. Specific architecture (see models.py), number of filters,kernel size, strides, loss function, L1 regularization can be defined in the config.

To train use the command:

```
python3 train.py -c quantized_cnn.yml
```

W.I.P: To evaluate performance:

```
python3 compare_models.py -c quantized_cnn/
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
