import numpy as np
import scipy
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, Conv2D, Flatten, Dense, UpSampling2D, Reshape, Concatenate, Dropout, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import keras
from keras import backend as K

# set to True to re-train CVAE
TRAIN = False

# define functions for processing images
def preprocess(_imgs):
    return (6 + np.sign(_imgs) * np.log10(abs(_imgs) + 1e-6))/6

def deprocess(_imgs):
    return 10**(6*_imgs - 6) - 1e-6

def get_alpha(epoch):
    # return 0.0
    return 0.004 * min(((epoch % 300)/300), 1.)
    # return 0.12 * min(((epoch % 300)/125), 1.)

# load kSZ data
kSZ_maps = np.load('kSZ_128.npy')
macc = np.load('macc_200c.npy')
m200c = np.load('m200c.npy')
macc[macc > 10] = 10
macc = np.repeat(macc, 3)
m200c = np.repeat(m200c, 3)


# split into train and test sets for training
sz_train, sz_test, mass_train, mass_test, macc_train, macc_test = train_test_split(np.expand_dims(kSZ_maps, axis=-1),
                                                                                   m200c,
                                                                                   macc,
                                                                                   test_size=0.15,
                                                                                   random_state=42)

X_TRAIN = [preprocess(sz_train), 10**(mass_train - 13), macc_train]
X_VAL = [preprocess(sz_test), 10**(mass_test - 13), macc_test]

# build CVAE network
latent_dim = 64
dim = 128

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    
e_mass_input = Input(shape=(1,))
e_mass_accr_input = Input(shape=(1,))

encoder_inputs = Input(shape=(dim, dim, 1))

x = Conv2D(64, (3, 3), strides=2, padding='same')(encoder_inputs)
x = LeakyReLU(alpha=.01)(x)
x = Conv2D(64, (3, 3), strides=2, padding='same')(x)
x = LeakyReLU(alpha=.01)(x)

x = Conv2D(128, (3, 3), strides=2, padding='same')(x)
x = LeakyReLU(alpha=.01)(x)
x = Conv2D(128, (3, 3), strides=2, padding='same')(x)
x = LeakyReLU(alpha=.01)(x)

x = Conv2D(256, (3, 3), strides=2, padding='same')(x)
x = LeakyReLU(alpha=.01)(x)
x = Conv2D(256, (3, 3), strides=2, padding='same')(x)
x = LeakyReLU(alpha=.01)(x)
x = Conv2D(256, (3, 3), strides=2, padding='same')(x)
x = LeakyReLU(alpha=.01)(x)

x = Flatten()(x)
x = Concatenate()([x, e_mass_input, e_mass_accr_input])

x = Dense(256)(x)
x = LeakyReLU(alpha=0.01)(x)
x = Dense(256)(x)
x = LeakyReLU(alpha=0.01)(x)

z_mean = Dense(latent_dim, name="z_mean")(x)
z_log_var = Dense(latent_dim, name="z_log_var")(x)

z = Sampling()([z_mean, z_log_var])

encoder = Model([encoder_inputs, e_mass_input, e_mass_accr_input], [z_mean, z_log_var, z], name="encoder")

encoder.summary()

latent_inputs = Input(shape=(latent_dim,))
d_mass_input = Input(shape=(1,))
d_mass_accr_input = Input(shape=(1,))

x = Concatenate()([latent_inputs, d_mass_input, d_mass_accr_input])
x = Dense(256)(x)
x = LeakyReLU(alpha=0.01)(x)

x = Reshape((1, 1, 256))(x)

x = Conv2D(256, (3, 3), padding='same')(x)
x = LeakyReLU(alpha=.01)(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(256, (3, 3), padding='same')(x)
x = LeakyReLU(alpha=.01)(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(256, (3, 3), padding='same')(x)
x = LeakyReLU(alpha=.01)(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(128, (3, 3), padding='same')(x)
x = LeakyReLU(alpha=.01)(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), padding='same')(x)
x = LeakyReLU(alpha=.01)(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(64, (3, 3), padding='same')(x)
x = LeakyReLU(alpha=.01)(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = LeakyReLU(alpha=.01)(x)
x = UpSampling2D((2, 2))(x)

decoder_outputs = Conv2D(1, (3, 3), activation=K.sigmoid, padding='same')(x)
decoder = tf.keras.Model([latent_inputs, d_mass_input, d_mass_accr_input], decoder_outputs, name="decoder")

# create VAE object
BATCH_SIZE = 8

alpha = K.variable(1)
beta = 1

val_losses = []

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.ones = K.ones((BATCH_SIZE, 128, 128, 1))
        self.steps = 0
    
    def rotate(self, x):
        # Rotate 0, 90, 180, 270 degrees
        return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

    def deprocess(self, _imgs):
        return 10**(6*tf.math.minimum(_imgs, self.ones) - 6) - 1e-6
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            imgs = self.rotate(data[0])

            maxes = tf.identity(K.reshape(K.max(K.reshape(imgs, shape=(BATCH_SIZE, dim*dim)), axis=1), shape=(BATCH_SIZE, 1, 1, 1)), name='maxes')
        
            z_mean, z_log_var, z = encoder([imgs, data[1], data[2]])

            reconstruction = decoder([z, data[1], data[2]])

            error = imgs-reconstruction

            reconstruction_loss = K.mean(K.square(error))
            reconstruction_loss *= dim*dim

            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.mean(kl_loss)
            kl_loss *= -0.5
                        
            total_loss = reconstruction_loss + alpha*kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }
    
    
    def test_step(self, data):
        batch_size = tf.shape(data[0])[0]

        normalized_data = data[0]
        z_mean, z_log_var, z = encoder([normalized_data, data[1], data[2]])
        reconstruction = decoder([z, data[1], data[2]])  
        
        error = data[0]-reconstruction

        reconstruction_loss = K.mean(K.square(error))
        reconstruction_loss *= dim*dim

        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.mean(kl_loss)
        kl_loss *= -0.5
        
        total_loss = reconstruction_loss + alpha*kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }
    
vae = VAE(encoder, decoder)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=0, vertical_flip=True, horizontal_flip=True,
#                                                           fill_mode='constant', cval=0
                                                         )
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=0, vertical_flip=True, horizontal_flip=True,
#                                                           fill_mode='constant', cval=0
                                                         )

class KLAnnealingCallback(tf.keras.callbacks.Callback):
    def __init__(self, schedule, variable):
        super(KLAnnealingCallback, self).__init__()
        self.schedule = schedule
        self.variable = variable
    def on_epoch_begin(self, epoch, logs={}):
        value = self.schedule(epoch)
        assert type(value) == float
        K.set_value(self.variable, value)

# train CVAE if training, otherwise just load saved weights
if TRAIN:
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_reconstruction_loss', patience=150)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_reconstruction_loss', factor=0.2, patience=75, verbose=1)
    nan_terminate = tf.keras.callbacks.TerminateOnNaN()
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join('unnorm_tmp', 'model_checkpoint.ckpt'),
        save_freq=5 * len(X_TRAIN[0]),
        save_weights_only=True,
        monitor='reconstruction_loss',
        save_best_only=True)

    logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    history = vae.fit(
        datagen.flow((X_TRAIN[0], X_TRAIN[1:]), batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_TRAIN[0]) / BATCH_SIZE,
        epochs=300,
        shuffle=True,
        validation_data=val_datagen.flow((X_VAL[0], X_VAL[1:]), batch_size=BATCH_SIZE),
        callbacks=[early_stopping, reduce_lr, nan_terminate, model_checkpoint_callback, KLAnnealingCallback(get_alpha, alpha)]
    )
    
    vae.save_weights('kSZ_vae_weights')
else:
    vae.load_weights("cvae_weights/kSZ_vae_weights")  



