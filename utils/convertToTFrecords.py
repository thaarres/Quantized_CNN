import tensorflow as tf
import h5py

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    

h5 = h5py.File('jetImage/jetimg/jetimg.h5', 'r')
h5.keys()
print(h5['y_train'][0], type(_float_feature(h5['y_train'][0])))
# X_train = h5['X_train']#[:60000]
# y_train = h5['y_train']#[:60000]
