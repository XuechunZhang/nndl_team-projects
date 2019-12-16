#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:49:31 2019

@author: xuechun
"""


from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf #2.0.0
tf.get_logger().setLevel('INFO') # disable warnings

from tensorflow.keras.layers import Input, Dense, Activation, AveragePooling2D,Flatten,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.datasets import cifar10

import tensorflow.keras.backend as BK

from MyLayer import ABCLayer

#%% load data
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

inputShape = x_train.shape[1:]

#%% Define ABC model paramenters
M = 5
num_activations = 3
shift_para = BK.random_uniform(shape=(num_activations, 1), minval=-1, maxval=1)
betas = BK.variable(BK.constant(1., shape=(num_activations, 1)), dtype="float32")
betas = BK.reshape(betas, shape=[num_activations] + [1] * (len(inputShape)+1))

#%% Define resnet layer based on ABC
def resnetABCLayer(inputs,
                   weight_initial,
                   shiftPara_initial,
                   beta_initial,
                   strides = 1,
                   activation = 'relu',
                   batch_normalization = True):
    '''
    conv + batchnorm + activation(relu)
    '''
    ABC = ABCLayer(weight_initial = weight_initial,
                   shiftPara_initial = shiftPara_initial,
                   beta_initial = beta_initial,
                   M = M,
                   N = num_activations,
                   strides=strides)
    
    x = inputs
    x = ABC(x)
    
    if batch_normalization:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)
    
    return x

#%% Construct ABC-based resnet blocks and assemble them using for loop
def resnet_trial(input_shape,
                 batch_size,
                 betas = betas,
                 M = M,
                 num_activations = num_activations,
                 num_filters = 16,
                 strides = 1,
                 num_classes = 10):
    
    # Define Define ABC model paramenters: W's shape
    conv_layer = BK.random_normal(shape=(3,3,3,num_filters))
    
    # input layer
    inputs = Input(shape=inputShape) 
    x = resnetABCLayer(inputs=inputs,
                       weight_initial = conv_layer,
                       shiftPara_initial = shift_para,
                       beta_initial = betas) # first resnet layer
    
    num_res_blocks = 3
    
    for stack in range(3):
        for res_block in range(num_res_blocks):            
            #construct one single resnet block
            strides_in = strides
            # Define Define ABC model paramenters: W's shape
            conv_layer = BK.random_normal(shape=(3,3,num_filters,num_filters))
            
            # y-path: first y
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides_in = 2  # downsample
                conv_layer = BK.random_normal(shape=(3,3,int(num_filters/2),num_filters))
            y = resnetABCLayer(inputs=x,
                               weight_initial = conv_layer,
                               shiftPara_initial = shift_para,
                               beta_initial = betas,
                               strides = strides_in)
            
            # y-path: second y
            if stack > 0 and res_block == 0:  # first layer but not first stack
                conv_layer = BK.random_normal(shape=(3,3,num_filters,num_filters))
            y = resnetABCLayer(inputs=y,
                               weight_initial = conv_layer,
                               shiftPara_initial = shift_para,
                               beta_initial = betas,
                               activation=None)
            
            # x-path
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # downsample
                conv_layer = BK.random_normal(shape=(3,3,int(num_filters/2),num_filters))
                strides_in = 2
                x = resnetABCLayer(inputs=x,
                                   weight_initial = conv_layer,
                                   shiftPara_initial = shift_para,
                                   beta_initial = betas,
                                   strides=strides_in,
                                   activation=None,
                                   batch_normalization=False)
            
            x = tf.keras.layers.add([x, y])
            x = Activation('relu')(x)
            print('graph built success',stack,res_block)
        num_filters *= 2
         
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)

    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # construct model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

#%% Fit model
batch_size = 32
epochs = 1
model = resnet_trial(input_shape=inputShape, batch_size = batch_size)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0001),
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)
    
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

#%% Save model
model.save('ResNet20_ABC_Keras.h5')