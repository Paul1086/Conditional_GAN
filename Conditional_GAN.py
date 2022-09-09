#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
from tensorflow import keras
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, Dropout
from keras.layers import Concatenate, BatchNormalization, Activation, Reshape, RepeatVector
from matplotlib import pyplot as plt
import numpy as np
import time
from IPython import display


# In[6]:


real_images = np.load("D:/IDL-Sudipta/Project/float16sata/new_real_images_16.npy", mmap_mode = 'c')
attributes = np.load('D:/IDL-Sudipta/Project/float16sata/new_attributes_16.npy', mmap_mode = 'c')


# In[7]:


attributes.shape


# In[8]:


class define_discriminator(Model):
    def __init__(self):
        super(define_discriminator, self).__init__()
        self.c1 = tf.keras.layers.Conv2D(64, (2, 2), strides=(2, 2), padding='same',input_shape=(64, 64, 3))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation('relu')

        self.c2 = tf.keras.layers.Conv2D(64, (2,2), strides=(2,2), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.Activation('relu')
    
        self.c3 = tf.keras.layers.Conv2D(256, (2,2), strides=(2,2), padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.act3 = tf.keras.layers.Activation('relu')
    
    
        self.c4 = tf.keras.layers.Conv2D(512, (2,2), strides=(2,2), padding='same')
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.act4 = tf.keras.layers.Activation('relu')
    
    
        self.c5 = tf.keras.layers.Conv2D(1024, (2,2), strides=(2,2), padding='same')
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.act5 = tf.keras.layers.Activation('relu')

        self.flatten = tf.keras.layers.Flatten()
        self.den1 = tf.keras.layers.Dense(1, activation='sigmoid')
    
    
    def call(self, data, label):
        
        data = self.c1(data)
        data = self.bn1(data)
        data = self.act1(data)
        
        
        label = label[:, None, None]
        label = tf.repeat(label, 64*64, axis=1)
        label = tf.reshape(label, (-1, 64, 64, 5))


        label = self.c2(label)
    
        data = tf.concat([data,label],axis=-1)
        data = self.c3(data)
        data = self.bn3(data)
        data = self.act3(data)
        
        data = self.c4(data)
        data = self.bn4(data)
        data = self.act4(data)
        
        data = self.c5(data)
        data = self.bn5(data)
        data = self.act5(data)
        

     
        data = self.flatten(data)
        data = self.den1(data)
        return data


# In[9]:


class define_generator(Model):
    def __init__(self):
        super().__init__()
        
        self.d1 = tf.keras.layers.Conv2DTranspose(512, (4, 4), strides=(1, 1), padding='valid')
        self.dbn1 = tf.keras.layers.BatchNormalization()
        self.dact1 = tf.keras.layers.Activation('relu')
        
        self.d2 = tf.keras.layers.Conv2DTranspose(512, (4, 4), strides=(1, 1), padding='valid')
        self.dbn2 = tf.keras.layers.BatchNormalization()
        self.dact2 = tf.keras.layers.Activation('relu')
        
        
        self.d3 = tf.keras.layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same')
        self.dbn3 = tf.keras.layers.BatchNormalization()
        self.dact3 = tf.keras.layers.Activation('relu')
        
        
        self.d4 = tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same')
        self.dbn4 = tf.keras.layers.BatchNormalization()
        self.dact4 = tf.keras.layers.Activation('relu')
        
        
        self.d5 = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')
        self.dbn5 = tf.keras.layers.BatchNormalization()
        self.dact5 = tf.keras.layers.Activation('relu')
        
        
        self.d6 = tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same')
        self.dact6 = tf.keras.layers.Activation('tanh')


    def call(self, data, label):
        
        data = data[:,None, None,:]
        data = self.d1(data)
        data = self.dbn1(data)
        data = self.dact1(data)
        
        label = label[:, None, None]
        label = self.d2(label)
        label = self.dbn2(label)
        label = self.dact2(label)
        
        data = tf.concat([data, label], axis = -1)
        data = self.d3(data)
        data = self.dbn3(data)
        data = self.dact3(data)
        
        data = self.d4(data)
        data = self.dbn4(data)
        data = self.dact4(data)
        
        data = self.d5(data)
        data = self.dbn5(data)
        data = self.dact5(data)
        
        data = self.d6(data)
        data = self.dact6(data)
        return data 


# In[14]:


EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 100
BUFFER_SIZE = 60000
BATCH_SIZE = 50


# In[7]:


train_dataset = tf.data.Dataset.from_tensor_slices((real_images[:200000,:,:,:], attributes[:200000,:])).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
discriminator = define_discriminator()
generator = define_generator()


# In[8]:


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# In[9]:


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# In[10]:


generator_optimizer = tf.keras.optimizers.Adam(0.0005)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0001)


# In[13]:


tf.random.set_seed(5)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# In[10]:


@tf.function
def train_step(images,y):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise,y, training=True)
        real_output = discriminator(images,y, training=True)
        fake_output = discriminator(generated_images,y, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# In[11]:


def train(dataset, epochs):
    for epoch in range(epochs):
        print(epoch)
        start = time.time()

        for image_batch,y in dataset:
            train_step(image_batch,y)
        display.clear_output(wait=True)
        generate_and_save_images(generator,y,
                             epoch + 1,
                             seed)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        m = "model_weights_IDL_16"+str(epoch)
        generator.save_weights(m)

    display.clear_output(wait=True)
    generate_and_save_images(generator,y,epochs,seed)


# In[13]:


def generate_and_save_images(model, y, epoch, test_input):
    y = attributes[:100]
    preds = model(test_input, y, training=False) 
    preds = (preds*127.5 + 127.5)/255
    fig = plt.figure(figsize = (10,10), dpi = 100)    
    for i in range(preds.shape[0]):
        plt.subplot(10,10,i+1)
        plt.imshow(preds[i,:,:,:])
        plt.title(str(i+1))
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


# In[17]:


train(train_dataset, EPOCHS)


# In[19]:


generator.save_weights("final_weights")


# In[ ]:




