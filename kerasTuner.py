import os, sys
import logging
import math
import h5py
import kerastuner as kt
import tensorflow as tf
logging.info('Tensorflow version ' + tf.__version__)
import tensorflow_datasets as tfds
AUTO = tf.data.experimental.AUTOTUNE
    
from utils.generator import DataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.get_logger().setLevel('ERROR')
    
def doTest():
  with strategy.scope(): # this line is all that is needed to run on TPU (or multi-GPU, ...)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(kernel_size=3, filters=16, padding='same', activation='relu', input_shape=[*image_size]),
        tf.keras.layers.Conv2D(kernel_size=3, filters=30, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Conv2D(kernel_size=3, filters=60, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Conv2D(kernel_size=3, filters=90, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(kernel_size=1, filters=40, padding='same', activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
      optimizer='adam',
      loss= 'categorical_crossentropy',
      metrics=['accuracy'])

    model.summary()
    history = model.fit(ds_train, steps_per_epoch=steps_per_epoch, epochs=5,
                      validation_data=ds_test)

def convert_dataset(item):
  '''Puts the mnist dataset in the format Keras expects, (features, labels).'''
  image = item['image']
  label = item['label']
  label = tf.one_hot(tf.squeeze(label), 10)
  image = tf.dtypes.cast(image, 'float32') / 255.
  return image, label

def getJetData2():
  x_train = tfio.IODataset.from_hdf5(path_hdf5_x_train, dataset='/X_train')
  y_train = tfio.IODataset.from_hdf5(path_hdf5_y_train, dataset='/y_train')

  # Zip together samples and corresponding labels
  ds_train = tf.data.Dataset.zip((x_train,y_train)).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
  ds_test = tf.data.Dataset.zip((x_train,y_train)).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
 
  
def getJetData():
  h5 = h5py.File('jetImage/jetimg/jetimg.h5', 'r')
  X_train = h5['X_train'][:60000]
  y_train = h5['y_train'][:60000]
  in_shape = X_train.shape[1:]
  train_size = X_train.shape[0]
  train_generator = DataGenerator(X_train, y_train,
                                  batch_size=batch_size,
                                  in_dim=in_shape,
                                  out_dim=(nclasses),
                                  shuffle=False
                                  )

  val_generator = DataGenerator(X_train, y_train,
                                batch_size=batch_size,
                                in_dim=in_shape,
                                out_dim=(nclasses),
                                shuffle=False,
                                validation=True
                                )
                                                              
  return train_generator,val_generator, train_size
        
def getData(dataset):
  
  if dataset.find('svhn')!=-1:
    '''Get the SVHN cropped dataset (32,32,3)'''
    ds_data, info = tfds.load('svhn_cropped', with_info=True, data_dir='/afs/cern.ch/user/t/thaarres/tensorflow_datasets/') 
    ds_train, ds_test = ds_data['train'], ds_data['test']
    ds_train = ds_train.map(convert_dataset)    
    # ds_train = ds_train.cache()
    ds_train = ds_train.repeat()
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(AUTO)
  
    ds_test  = ds_test.map(convert_dataset)
    # ds_test  = ds_test.cache()
    ds_test  = ds_test.batch(batch_size)
    ds_test  = ds_test.prefetch(AUTO)
    
    train_size = int(info.splits['train'].num_examples)
  
  else:
    '''Get the jet image dataset (100,100,1)'''
    ds_train, ds_test, train_size = getJetData()
  
  return ds_train, ds_test, train_size



def build_model(hp):
  '''Define the model to test'''
  inputs = tf.keras.Input(shape=[*image_size])
  x = inputs
  for i in range(hp.Int('conv_layers', 1, 4, default=3)):
    x = tf.keras.layers.Conv2D( #tf.keras.layers.SeparableConv2D(
        filters=hp.Choice('filters_' + str(i), [4,8,16,32,64]),
        # filters=hp.Int('filters_' + str(i), 4, 32, step=4, default=8),
        kernel_size=hp.Int('kernel_size_' + str(i), 3, 5),
        padding='same')(x)

    if hp.Choice('pooling' + str(i), ['max', 'avg']) == 'max':
        x = tf.keras.layers.MaxPooling2D()(x)
    else:
        x = tf.keras.layers.AveragePooling2D()(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

  if hp.Choice('global_flatten', ['max', 'avg','flatten']) == 'max':
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
  elif hp.Choice('global_flatten', ['max', 'avg','flatten']) == 'avg':
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
  else:
    x = tf.keras.layers.Flatten()(x)
        
  for i in range(hp.Int('dense_layers', 0, 2, default=1)):
    x = tf.keras.layers.Dense(units=hp.Int('neurons_' + str(i), 16, 128, step=16, default=32))(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout( rate=hp.Choice('droupout_rate', [0.,0.2, 0.5, 0.6,0.8]))(x)

  outputs = tf.keras.layers.Dense(nclasses, activation='softmax')(x)
  model = tf.keras.Model(inputs, outputs)

  # lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log', default=1e-3)
  model.compile(tf.keras.optimizers.Adam(0.002), loss='categorical_crossentropy', metrics=['accuracy'])
  return model

if __name__ == '__main__':
  
  dataset  = 'svhn'
  epochs   = 60
  nclasses = 10
  batch_size = 32
  
  if len(sys.argv) > 1:
    dataset = str(sys.argv[1])   
  if len(sys.argv) > 2:
    epochs = int(sys.argv[2])    
  
  if dataset.find('jetimages')!=-1:
    nclasses = 5
    
  logging.getLogger().setLevel(logging.INFO)
  
  logging.basicConfig(filename='kerasTuner.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    
  logging.info('Set up accelerators (if available)')
  try:
    tpu = None
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
  except ValueError:
    strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines
  logging.info(f'{strategy.num_replicas_in_sync} accelerators')
  if tpu:
    batch_size = batch_size*strategy.num_replicas_in_sync
  
  
  logging.info(f'Getting dataset: {dataset}')  
  ds_train, ds_test, train_size = getData(dataset)
  for img_feature, label in ds_train:
        break
  logging.info(f'Input shape (batch, height, width, channels) = {img_feature.shape}')
  logging.info(f'Label shape (batch, n classes) = {label.shape}')
  image_size = img_feature.shape[1:]
  logging.info(f'Image size is = {image_size}')
  logging.info(f'Training on {train_size} events')

  steps_per_epoch=train_size//batch_size
  logging.info(f'Using N steps per epoch N = {steps_per_epoch}')
  
  # doTest()# For testing
  tuner = kt.Hyperband(
        hypermodel=build_model,
        objective='val_accuracy',
        max_epochs=100,
        factor=3,
        hyperband_iterations=3,
        distribution_strategy=strategy,
        directory='/afs/cern.ch/work/t/thaarres/public/kerasTune/%s/'%dataset,
        project_name='v4_dropout',
        overwrite=False)
  tuner.search_space_summary()
  logging.info('Start search')

  if dataset == 'svhn':
    logging.info('Tuning SVHN model ')
    tuner.search(ds_train,
               steps_per_epoch=steps_per_epoch,
               validation_data=ds_test,
               epochs=epochs,
               callbacks=[tf.keras.callbacks.EarlyStopping('val_accuracy',patience=1)])#,ClearTrainingOutput()])
  elif dataset == 'jetimages':
    logging.info('Tuning jet images model')
    tuner.search(ds_train,
              steps_per_epoch=len(ds_train),
              epochs=epochs,
              validation_data=ds_test,
              validation_steps=len(ds_test),
              callbacks=[tf.keras.callbacks.EarlyStopping('loss',patience=1)])
  else:
    logging.error("Invalid dataset!")
  tuner.results_summary()
  logging.info('Get the optimal hyperparameters')
  best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

  logging.info(f'''
  The hyperparameter search is complete. The optimal number of conv layers is {best_hps.get('conv_layers')} and the optimal learning rate for the optimizer
  is {best_hps.get('learning_rate')}.
  ''')

  logging.info('Retrain using the best model!')
  # Build the model with the optimal hyperparameters and train it on the data
  model = tuner.hypermodel.build(best_hps)
  model.summary()
  if dataset.find('svhn')!=-1:
    model.fit(ds_train,
              steps_per_epoch=steps_per_epoch,
              validation_data=ds_test,
              epochs=100,
              callbacks=[tf.keras.callbacks.EarlyStopping('val_accuracy',patience=1)])
  else:
    model.fit(ds_train,
              steps_per_epoch=len(ds_train),
              epochs=100,
              validation_data=ds_test,
              validation_steps=len(ds_test),
              callbacks=[tf.keras.callbacks.EarlyStopping('loss',patience=1)])