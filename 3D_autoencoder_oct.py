#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 10:09:45 2022

@author: diego
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 11:32:19 2022

@author: diego
"""

from glob import glob
import random
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses

from datetime import datetime

datetime.now()




def make_dataset(path,target_path, batch_size):
    @tf.function
    def parse_volume_tf(filename):
        def normalize(volume):
            """Normalize the volume"""
            min = 0
            max = 255
            volume[volume < min] = min
            volume[volume > max] = max
            volume = (volume - min) / (max - min)
            volume = volume.astype("float32")
            return volume


        def process_scan(volume):
            """Read and resize volume"""
            # Normalize
            volume = normalize(volume)

            return volume
        
        

        def parse_volume(filename):

            with open( filename.numpy(), 'rb') as f:
                volume= pickle.load(f)
            
            volume = process_scan(volume)
            volume = tf.expand_dims(volume, axis=3)
            return volume
          
        
        return tf.py_function(parse_volume,inp=[filename], Tout=tf.float32)
        
    def configure_for_performance(ds):
        #ds = ds.shuffle(buffer_size=50)
        ds = ds.batch(batch_size)
        #ds = ds.repeat()
        ds = ds.prefetch(1)
        return ds    
       
   
    filenames = glob(path+"/*.pkl", recursive = True)
    filenames.sort()
    
    target_filenames = glob(target_path+"/*.pkl", recursive = True)
    target_filenames.sort()
    
    ziped_filenames=zip(filenames, target_filenames)
    ziped_filenames = list(ziped_filenames)
    random.shuffle(ziped_filenames)
    ziped_filenames=list(zip(*ziped_filenames))

    
      
    filenames_ds = tf.data.Dataset.from_tensor_slices(list(ziped_filenames[0]))
    target_filenames_ds = tf.data.Dataset.from_tensor_slices(list(ziped_filenames[1]))
    
    volume_ds = filenames_ds.map(parse_volume_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    target_volume_ds = target_filenames_ds.map(parse_volume_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    ds = tf.data.Dataset.zip((volume_ds, target_volume_ds))
    ds = configure_for_performance(ds)
      
    return ds

path_train='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/subsampled_sub_volumes/subsampled_75/train'
target_path_train='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/subsampled_sub_volumes/original_75/train'

batch_size=16
data_set=make_dataset(path_train,target_path_train, batch_size)

path_test='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/subsampled_sub_volumes/subsampled_75/test'
target_path_test='/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/subsampled_sub_volumes/original_75/test'

test_data_set=make_dataset(path_test,target_path_test, batch_size)









class Denoise(Model):
  def __init__(self):
    super(Denoise, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(64, 64, 100,1)),
      layers.Conv3D(16, 3, activation='relu', padding='same', strides=2),
      layers.Conv3D(8, 3, activation='relu', padding='same', strides=2)])

    self.decoder = tf.keras.Sequential([
      layers.Conv3DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv3DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv3D(1, kernel_size=3, activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded



autoencoder = Denoise()
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())


EPOCHS = 10
total_train=len(data_set)*batch_size
total_val=len(test_data_set)*batch_size


history = autoencoder.fit(
    data_set,
    steps_per_epoch=int(np.ceil(total_train / float(batch_size))),
    epochs=EPOCHS,
    validation_data=test_data_set,
    validation_steps=int(np.ceil(total_val / float(batch_size)))
)


autoencoder.save('/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/models/model_75_'+str(datetime.now()))




autoencoder.encoder.summary()
autoencoder.decoder.summary()



# data = test_data_set.take(1)
# volume,target_volume= list(data)[0]
# volume=volume.numpy()
# target_volume=target_volume.numpy()
# plt.imshow(np.squeeze(volume[0,:,:,90,0]), cmap="gray")
# plt.imshow(np.squeeze(target_volume[0,:,:,90,0]), cmap="gray")

# encoded_imgs = autoencoder.encoder(volume).numpy()
# decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()


# plt.imshow(np.squeeze(decoded_imgs[0,:,:,90,0]), cmap="gray")
from tensorflow import keras
autoencoder = keras.models.load_model('/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/models/model_75')

kernel_and_biases_weights=[]
for super_layers in autoencoder.layers:
    for sub_layers in super_layers.layers:
        kernel_and_biases_weights.append(sub_layers.get_weights())
        


class Denoise_for_reconstruction(Model):
  def __init__(self):
    super(Denoise_for_reconstruction, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(512,400, 100,1)),
      layers.Conv3D(16, 3, activation='relu', padding='same', strides=2,use_bias=True,weights=kernel_and_biases_weights[0]),
      layers.Conv3D(8, 3, activation='relu', padding='same', strides=2,use_bias=True,weights=kernel_and_biases_weights[1])])

    self.decoder = tf.keras.Sequential([
      layers.Conv3DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same',use_bias=True,weights=kernel_and_biases_weights[2]),
      layers.Conv3DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same',use_bias=True,weights=kernel_and_biases_weights[3]),
      layers.Conv3D(1, kernel_size=3, activation='sigmoid', padding='same',use_bias=True,weights=kernel_and_biases_weights[4])])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded




# Denoise_for_reconstruction.build()

reconstruction_model=Denoise_for_reconstruction()


def load_obj(name):
    with open( name, 'rb') as f:
        return pickle.load(f)
    
sub_sampled_volume=load_obj('/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/sub_sampled_data/subsampled_75/test/Farsiu_Ophthalmology_2013_AMD_Subject_1001_subsampled_75.pkl')
original_volume=load_obj('/home/diego/Documents/Delaware/tensorflow/training_3D_images/subsampling/sub_sampled_data/original_75/test/Farsiu_Ophthalmology_2013_AMD_Subject_1001.pkl')

sub_sampled_volume=np.expand_dims(sub_sampled_volume[:,0:400,:], axis=0)
sub_sampled_volume=np.expand_dims(sub_sampled_volume, axis=4)


encoded_volume = reconstruction_model.encoder(sub_sampled_volume).numpy()
decoded_volume = reconstruction_model.decoder(encoded_volume).numpy()
decoded_volume=np.squeeze(decoded_volume)



import cv2
decoded_volume=decoded_volume*255
decoded_volume=decoded_volume.astype('uint8')


cv2.imshow('Reconstructed Volume',decoded_volume[:,:,0])
cv2.waitKey(0)
cv2.destroyAllWindows()

