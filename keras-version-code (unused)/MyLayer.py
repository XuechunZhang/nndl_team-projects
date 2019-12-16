# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:29:55 2019

@author: Han_PC
"""

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer, multiply
from tensorflow.keras.initializers import Initializer
import tensorflow.keras.backend as BK


#%% ABC convolution related functions
def calculate_binary_weights(conv_layer, M):
    '''
    conv_layer: original layer's W
    '''
    mean = BK.mean(BK.reshape(conv_layer, shape=(-1,)), axis=0)
    variance = BK.var(BK.reshape(conv_layer, shape=(-1,)), axis=0)

    shifted_stddev = -1 + np.array(range(M)) * (2 / (M-1))
    shifted_stddev = BK.constant(shifted_stddev, dtype="float32", name="shifted_stddev")
    shifted_stddev *= BK.sqrt(variance)
    shifted_stddev = BK.reshape(shifted_stddev,
                                shape=[M] + [1] * len(conv_layer.get_shape()))
    binary_weights = conv_layer - mean
    binary_weights = BK.tile(BK.expand_dims(binary_weights, 0),
                             n=[M] + [1] * len(conv_layer.get_shape()))
    binary_weights = BK.sign(binary_weights + shifted_stddev)
    return binary_weights

def calculate_alphas(conv_layer, binary_weights, M, num_epoc=500, learning_rate=0.01):
    '''
    calculate alpha based on OLS
    '''
    flat_conv_layer = BK.reshape(conv_layer, shape=(np.prod(conv_layer.get_shape()),1))
    flat_binary_weights = BK.reshape(binary_weights, shape=(M,-1))
    alphas = BK.random_normal(shape=(M,1))

    flat_approx_layer = BK.dot(BK.transpose(flat_binary_weights), alphas)
    loss = []
    for _ in range(num_epoc):
        flat_approx_layer = BK.dot(BK.transpose(flat_binary_weights), alphas)
        curr_loss = BK.mean(BK.square(flat_approx_layer - flat_conv_layer), axis=0) #tf.reduce_mean, is that correct
        loss.append(curr_loss)
        grad = BK.dot(flat_binary_weights, (flat_approx_layer - flat_conv_layer))/flat_conv_layer.shape[0]
        alphas = alphas - learning_rate * grad
    return alphas, loss

def approx_conv_layer(inputs, binary_weights, alphas, M, strides=1, padding="same"):
    '''
    approximation step
    '''
    expanded_alphas = BK.reshape(alphas, shape=[M] + [1] * len(inputs.get_shape()))
    approx_conv_layer = []
    for idx in range(M):
        curr_layer = BK.conv2d(inputs, binary_weights[idx], strides=strides, padding=padding,data_format = "channels_last")
        approx_conv_layer.append(curr_layer)
    approx_layer = tf.convert_to_tensor(approx_conv_layer, dtype="float32", name="approx_layer")

    approx_out = BK.sum(multiply([approx_layer, expanded_alphas]), axis=0)
    return approx_out


#%% Define ABCLayer to replace Conv2D layer

class MyInitializer(Initializer):
    '''
    initializer: initialize variable to given values
    '''
    def __init__(self, values):
        self.values = values
    
    def __call__(self, shape, dtype="float32"):
        return BK.variable(self.values, dtype = dtype)



class ABCLayer(Layer):
    def __init__(self,
                 weight_initial,
                 shiftPara_initial,
                 beta_initial,
                 M,
                 N,
                 strides=1,
                 padding="same",
                 **kwargs):
        
        super(ABCLayer,self).__init__(**kwargs)
        
        self.weight_initial = weight_initial
        self.shiftPara_initial = shiftPara_initial
        self.beta_initial = beta_initial
        self.M = M
        self.N = N
        self.strides = strides
        self.padding = padding
    
    def build(self,input_shapes):
        self.weight = self.add_weight(shape = self.weight_initial.shape,
                                     initializer = MyInitializer(self.weight_initial),
                                     name='weight',
                                     trainable = True)
        self.shiftPara = self.add_weight(shape = self.shiftPara_initial.shape,
                                      initializer = 'zeros',
                                      name='shiftPara',
                                      trainable = True)
        self.beta = self.add_weight(shape = self.beta_initial.shape,
                                      initializer = 'ones',
                                      name='beta',
                                      trainable = True)
        
        self.built = True
    
    def call(self, inputs):  
        
        weight = self.weight
        shiftPara = self.shiftPara
        beta = self.beta
        binary_weights = calculate_binary_weights(weight, M = self.M)
        alphas, _ = calculate_alphas(weight, binary_weights, M = self.M)
        
        # approximate original convolution layer based on trainable parameters above
        activation_layer = []
        for idx in range(self.N):
            shifted_inputs = BK.clip(inputs + shiftPara[idx], 0., 1.)
            binary_inputs = BK.sign(shifted_inputs - 0.5)
            activation_layer.append(approx_conv_layer(binary_inputs, binary_weights, alphas, M=self.M, strides=self.strides, padding=self.padding))
        activation_layer = tf.convert_to_tensor(activation_layer, dtype="float32", name="activation_layer")

        activation_out = BK.sum(multiply([activation_layer, beta]), axis=0)

        return activation_out
        


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        