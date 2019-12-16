# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 14:03:47 2019

@author: Han_PC
"""
import tensorflow as tf #2.0.0
tf.get_logger().setLevel('INFO')

from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, AveragePooling2D,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10

#%% Read in data
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

inputShape = x_train.shape[1:]

#%% Define resnet layer
def resnetLayer(inputs,
                num_filters = 16,#Cout
                kernel_size = (3,3),
                strides = 1,
                activation = 'relu',
                batch_normalization = True):
    '''
    conv + batchnorm + activation(relu)
    '''
    conv = Conv2D(num_filters,
                  kernel_size = kernel_size,
                  strides = strides,
                  padding = 'same',
                  kernel_initializer = 'he_normal',
                  kernel_regularizer = l2(1e-4))
    
    x = inputs
    x = conv(x)
    if batch_normalization:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)
    
    return x


#%% Construct resnet blocks and assemble them using for loop
def resnet20(input_shape,
           num_filters = 16,
           strides = 1,
           kernel_size = (3,3),
           num_classes = 10):
    
    num_res_blocks = 3
    
    inputs = Input(shape=inputShape) # input layer
    x = resnetLayer(inputs) # first resnet layer
    
    for stack in range(3):
        for res_block in range(num_res_blocks):
            '''
            construct one single resnet block
            '''
            #print(stack,res_block)
            strides_in = strides
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides_in = 2  # down sample
            # y-path: first y
            y = resnetLayer(inputs=x,
                            num_filters=num_filters,
                            strides=strides_in)
            # y-path: second y
            y = resnetLayer(inputs=y,
                            num_filters=num_filters,
                            activation=None)
            # x-path
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # down sample
                x = resnetLayer(inputs=x,
                                num_filters=num_filters,
                                kernel_size=kernel_size,
                                strides=strides_in,
                                activation=None,
                                batch_normalization=False)

            x = tf.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2
        
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # construct model
    model = Model(inputs=inputs, outputs=outputs)
    return model
    

#%% Fit model
model = resnet20(input_shape=inputShape)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])

batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 1
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)
    
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

#%% Save model
model.save('ResNet20_Keras.h5')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
