import tensorflow as tf
import numpy as np
import math

class DataGenerator(tf.compat.v2.keras.utils.Sequence):

    def __init__(self, X_data, y_data, batch_size, in_dim, out_dim,
                 shuffle=True, validation=False, val_size=0.2):

        self.batch_size = batch_size
        self.X_data = X_data
        self.y_data = y_data
        self.n_classes = out_dim
        self.dim = in_dim
        self.shuffle = shuffle
        self.n = 0

        size = len(self.X_data)
        train_size = 1 - val_size
        if validation:
            self.indices = np.arange(math.ceil(size*train_size))
        else:
            self.indices = np.arange(math.ceil(size*train_size), size)

        self.on_epoch_end()

    def __next__(self):
        data = self.__getitem__(self.n)
        self.n += 1
        if self.n >= self.__len__():
            self.on_epoch_end
            self.n = 0
        return data

    def __len__(self):
        return math.ceil(len(self.indices)/self.batch_size)

    def __getitem__(self, index):
        bx = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X = self._generate_x(bx)
        y = self._generate_y(bx)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _generate_x(self, indices):
        X = np.empty((self.batch_size, *self.dim))
        for i, d in enumerate(indices):
            X[i] = self.X_data[d]
        return X

    def _generate_y(self, indices):
        y = np.empty((self.batch_size, self.n_classes))
        for i, d in enumerate(indices):
            y[i] = self.y_data[d]
        return y