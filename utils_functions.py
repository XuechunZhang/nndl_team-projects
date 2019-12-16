import tensorflow as tf
import numpy as np

### basic utils functions
def weight_variable(shape, name="weight"):
    '''
    create convolution layer weight variable given shape
    '''
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

def bias_variable(shape, name="bias"):
    '''
    create convolution layer bias variable given shape
    '''
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)

def conv2d(inputs, weights, strides = [1,1,1,1]):
    '''
    call 2D-convolution function
    '''
    return tf.nn.conv2d(inputs, weights, strides, padding='SAME')

### ABC layer functions
def calculate_binary_weights(conv_W, M):
    '''
    Calculate M binary weights given convolution layer weights
    input:
    conv_W has shape (kernel_width, kernel_height, c_in, c_out)
    M is a scalar
    output:
    binary_weights has shape (M, kernel_width, kernel_height, c_in, c_out)
    '''
    with tf.name_scope("calculate_binary_weights"):
        # calculate mean and variance
        mean, variance = tf.nn.moments(tf.reshape(conv_W, shape=(-1,)), axes=0)
        # calculate shifted standard deviation
        if M > 1:
            shifted_stddev = (-1 + np.array(range(M)) * (2 / (M-1))) * tf.sqrt(variance)
            shifted_stddev = tf.reshape(shifted_stddev,
                                        shape=[M] + [1] * len(conv_W.get_shape()),
                                        name="shifted_stddev")
        else:
            shifted_stddev = tf.reshape([0],
                                        shape=[M] + [1] * len(conv_W.get_shape()),
                                        name="shifted_stddev")
        # reduce mean
        binary_weights = conv_W - mean
        # expand dimension
        binary_weights = tf.tile(tf.expand_dims(binary_weights, 0),
                                 multiples=[M] + [1] * len(conv_W.get_shape()))
        # add shifted standard deviation to get binary weights
        binary_weights = tf.sign(binary_weights + shifted_stddev, name="binary_weights")
        return binary_weights

def calculate_alphas(conv_W, binary_weights, alphas, M):
    '''
    Get alphas to best approximate convolution layer weights with M binary weights
    input:
    conv_W has shape (kernel_width, kernel_height, c_in, c_out)
    binary_weights has shape (M, kernel_width, kernel_height, c_in, c_out)
    alphas has shape (M,)
    M is a scalar
    out:
    alpha_loss
    alpha_training
    '''
    with tf.name_scope("calculate_alphas"):
        # flaten convolution layer weights
        flat_conv_weights = tf.reshape(conv_W, shape=(-1,), name="flat_conv_weights")
        # flaten binary weights
        flat_binary_weights = tf.reshape(binary_weights, shape=(M,-1), name="flat_binary_weights")
        # approximated flatened convolution layer
        flat_approx_layer = tf.reduce_sum(tf.multiply(flat_binary_weights, alphas), axis=0, name="flat_approx_layer")
        # calculate loss
        alpha_loss = tf.reduce_mean(tf.square(flat_approx_layer - flat_conv_weights), axis=0, name='alpha_loss')
        # optimize alphas using Adam
        alpha_training = tf.train.AdamOptimizer().minimize(alpha_loss, var_list=[alphas], name="alphas_training")
        return alpha_loss, alpha_training

def approx_conv_layer(inputs, binary_weights, alphas, M, conv_bias=None, strides=1, padding="SAME"):
    '''
    Get the output of the approximated convolution layer
    input:
    inputs has shape (num_batch, width, height, c_in) and is the inputs of convolution layer
    binary_weights has shape (M, kernel_width, kernel_height, c_in, c_out)
    alphas has shape (M,)
    M is a scalar
    conv_bias has shape (c_out,) if specified
    output:
    approx_out has shape (num_batch, width, height, c_out) assuming padding='same'
    '''
    with tf.name_scope("approx_conv_layer"):
        # set bias to be zero if not specified
        if conv_bias is None:
            conv_bias = 0.
        strides_in = [1, strides, strides, 1]
        # expand dimension of alphas for multiplication
        expanded_alphas = tf.reshape(alphas, shape=[M] + [1] * len(inputs.get_shape()), name="expanded_alphas")
        # store outputs of M binary layers
        binary_layer_out = []
        for idx in range(M):
            # binary layer
            curr_layer = tf.nn.conv2d(inputs, binary_weights[idx], strides=strides_in, padding=padding)
            binary_layer_out.append(curr_layer + conv_bias)
        binary_layer_out = tf.convert_to_tensor(binary_layer_out, dtype=tf.float32, name="binary_layer_out")
        # linear combination of M binary outputs by alphas
        approx_out = tf.reduce_sum(tf.multiply(binary_layer_out, expanded_alphas), axis=0, name="approx_out")
        return approx_out

def ABC_layer(inputs, binary_weights, alphas, shift_para, betas, M, N, conv_bias=None, strides=1, padding="SAME"):
    '''
    Get the output of the ABC layer
    input:
    inputs has shape (num_batch, width, height, c_in) and is the inputs of convolution layer
    binary_weights has shape (M, kernel_width, kernel_height, c_in, c_out)
    alphas has shape (M,)
    shift_para has shape (N,)
    betas has shape (N,)
    M is a scalar
    N is a scalar
    conv_bias has shape (c_out,) if specified
    output:
    ABC_out has shape (num_batch, width, height, c_out) assuming padding='SAME'
    '''
    with tf.name_scope("ABC_layer"):
        # expand dimension of betas for multiplication
        expanded_betas = tf.reshape(betas, shape=[N] + [1] * len(inputs.get_shape()), name="expanded_betas")
        # store outputs of N binary activation layers
        activation_layers = []
        for idx in range(N):
            # shift the inputs and turn to binary inputs
            shifted_inputs = tf.clip_by_value(inputs + shift_para[idx], 0., 1.)
            binary_inputs = tf.sign(shifted_inputs - 0.5)
            # pass into the approximated convolution layer
            activation_layers.append(approx_conv_layer(binary_inputs, binary_weights, alphas, M, conv_bias, strides, padding))
        activation_layers = tf.convert_to_tensor(activation_layers, dtype=tf.float32, name="activation_layers")
        ABC_out = tf.reduce_sum(tf.multiply(activation_layers, expanded_betas), axis=0, name="ABC_out")
        return ABC_out

def ABC_no_activation(inputs, binary_weights, alphas, M, conv_bias=None, strides=1, padding="SAME"):
    '''
    Get the output of the ABC layer without binary activation
    input:
    inputs has shape (num_batch, width, height, c_in) and is the inputs of convolution layer
    binary_weights has shape (M, kernel_width, kernel_height, c_in, c_out)
    alphas has shape (M,)
    M is a scalar
    conv_bias has shape (c_out,) if specified
    output:
    ABC_no_activation_out has shape (num_batch, width, height, c_out) assuming padding='SAME'
    '''
    with tf.name_scope("ABC_layer"):
        # set bias to be zero if not specified
        if conv_bias is None:
            conv_bias = 0.
        strides_in = [1, strides, strides, 1]
        # expand dimension of alphas for multiplication
        expanded_alphas = tf.reshape(alphas, shape=[M] + [1] * len(inputs.get_shape()), name="expanded_alphas")
        # store outputs of M binary convolution layers
        ABC_no_activation_layers = []
        for idx in range(M):
            # pass into the binary convolution layer
            curr_layer = tf.nn.conv2d(inputs, binary_weights[idx], strides=strides_in, padding=padding)
            ABC_no_activation_layers.append(curr_layer + conv_bias)
        ABC_no_activation_layers = tf.convert_to_tensor(ABC_no_activation_layers, dtype=tf.float32, name="ABC_no_activation_layers")
        ABC_no_activation_out = tf.reduce_sum(tf.multiply(ABC_no_activation_layers, expanded_alphas), axis=0, name="ABC_no_activation_out")
        return ABC_no_activation_out

