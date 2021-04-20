# -*- coding: utf-8 -*-
"""
Basic layers implemented by tensorflow modules
"""

import tensorflow as tf
from tensorflow import nn

from tensorflow.layers import Conv2D
from tensorflow.layers import Dense
from tensorflow.layers import BatchNormalization
from tensorflow.layers import MaxPooling2D
from tensorflow.layers import AveragePooling2D
from tensorflow.layers import Flatten
from tensorflow.layers import Dropout

from tensorflow.python.ops import init_ops


# ---------------
# layers implemented by tensorflow.layers
# ---------------
def conv(inputs, out_filters, ksize=(3, 3), strides=(1, 1), dilation=(1, 1),
         use_bias=True):
    """
    Convolution layer

    Parameters
    ----------
    inputs: Input tensor
    out_filters: Number of output filters
    ksize: Kernel size. One integer of tuple of two integers
    strides: Strides for moving kernel. One integer of tuple of two integers
    dilation: Dilation of kernel. One integer of tuple of two integers
    use_bias: Whether to use bias
    """
    return Conv2D(filters=out_filters,
                  kernel_size=ksize,
                  strides=strides,
                  padding='same',
                  dilation_rate=dilation,
                  use_bias=use_bias,
                  kernel_initializer=init_ops.glorot_uniform_initializer,
                  )(inputs)


def dense(inputs, out_length, use_bias=True):
    """
    Dense connected layer

    Parameters
    ----------
    inputs: Input tensor
    out_length: Length of outputs
    use_bias: Whether to use bias
    """
    return Dense(units=out_length,
                 use_bias=use_bias,
                 kernel_initializer=init_ops.glorot_uniform_initializer,
                 )(inputs)


def batch_normalization(inputs, training=False):
    """
    Batch normalization

    Parameters
    ----------
    inputs: Input tensor
    training: Whether in training phase
    """
    return BatchNormalization()(inputs, training=training)


def maxpool(inputs, pool_size=(2, 2), strides=(2, 2)):
    """
    Max pool

    Parameters
    ----------
    inputs: Input tensor
    pool_size: Size of the pooling window. One integer of tuple of two
        integers, (height_factor, width_factor)
    strides: Strides of the pooling operation. One integer of tuple of two
        integers
    """
    return MaxPooling2D(pool_size=pool_size, strides=strides)(inputs)


def average_pool(inputs, pool_size=(2, 2), strides=(2, 2)):
    """
    Average pool

    Parameters
    ----------
    inputs: Input tensor
    pool_size: Size of the pooling window. One integer of tuple of two
        integers, (height_factor, width_factor)
    strides: Strides of the pooling operation. One integer of tuple of two
        integers
    """
    return AveragePooling2D(pool_size=pool_size, strides=strides)(inputs)


def global_average_pool(inputs):
    """
    Global average pool, the height and width dimension will be squeezed, that
    is, the dims of output tensor will be (batch, out_filters)

    Parameters
    ----------
    inputs: Input tensor
    """
    return tf.reduce_mean(inputs, axis=[1, 2])


def flatten(inputs):
    """
    Flattens the inputs

    Parameters
    ----------
    inputs: Input tensor
    """
    return Flatten()(inputs)


def dropout(inputs, drop_rate):
    """
    Applies Dropout to the input

    Parameters
    ----------
    inputs: Input tensor
    drop_rate: float between 0 and 1. Fraction of the input units to drop.
    """
    return Dropout(rate=drop_rate)(inputs)


# ---------------
# layers implemented by basic tensorflow modules
# ---------------
def relu(inputs):
    """
    Relu activation

    Parameters
    ----------
    inputs: Input tensor
    """
    return nn.relu(inputs)


def leaky_relu(inputs, alpha=0.2):
    """
    Leaky relu activation

    Parameters
    ----------
    inputs: Input tensor
    alpha: Slope of negative neurons
    """
    return nn.leaky_relu(inputs, alpha=alpha)


def softmax(inputs, axis=-1):
    """
    Softmax activation

    Parameters
    ----------
    inputs: Input tensor
    axis: Axis along which the softmax normalization is applied
    """
    return nn.softmax(inputs, axis=axis)


def concat(inputs_list, axis=-1):
    """
    Concatenate input tensors list

    Parameters
    ----------
    inputs_list: Input tensors list
    axis: Axis along which to concatenate the tensors
    """
    return tf.concat(inputs_list, axis=axis)


def add(inputs_list):
    """
    Add input tensors list

    Parameters
    ----------
    inputs_list: Input tensors list
    """
    inputs_num = len(inputs_list)
    if inputs_num < 2:
        raise ValueError('number of tensors in inputs_list must be >= 2')

    outputs = inputs_list[0]
    for i in range(1, inputs_num):
        outputs += inputs_list[i]
    return outputs


def multiply(inputs_list):
    """
    Multiply input tensors list

    Parameters
    ----------
    inputs_list: Input tensors list
    """
    inputs_num = len(inputs_list)
    if inputs_num < 2:
        raise ValueError('number of tensors in inputs_list must be >= 2')

    outputs = inputs_list[0]
    for i in range(1, inputs_num):
        outputs *= inputs_list[i]
    return outputs


def reshape(inputs, new_shape):
    """
    Reshapes an output to a certain shape.

    Parameters
    ----------
    inputs: Input tensor
    new_shape: New shape without batch_size
    """
    return tf.reshape(inputs, shape=new_shape)


def permute(inputs, new_dims):
    """
    Permutes the dimensions of the input according to a given pattern

    Parameters
    ----------
    inputs: Input tensor
    new_dims: Re-ordered dimensions
    """
    return tf.transpose(inputs, perm=new_dims)


def upsample(inputs, factor=(2, 2), interpolation='nearest'):
    """
    Upsampling layer by factor

    Parameters
    ----------
    inputs: Input tensor
    factor: The upsampling factors for (height, width). One integer or tuple of
        two integers
    interpolation: A string, one of [`nearest`, `bilinear`, 'bicubic', 'area'].
    """
    # get new_size
    _, height, width, _ = inputs.get_shape().as_list()
    factor = _make_pair(factor)
    new_height = height * factor[0]
    new_width = width * factor[1]
    new_size = (new_height, new_width)

    # get interpolation type
    interp_types = {
        'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        'bilinear': tf.image.ResizeMethod.BILINEAR,
        'bicubic': tf.image.ResizeMethod.BICUBIC,
        'area': tf.image.ResizeMethod.AREA,
    }
    if interpolation not in interp_types.keys():
        raise ValueError("interpolation must be one of "
                         "['nearest', 'bilinear', 'bicubic', 'area']")
    interp_type = interp_types.get(interpolation)

    return tf.image.resize_images(inputs, size=new_size, method=interp_type)


def instance_normalization(inputs):
    """
    Instance normalization layer

    Parameters
    ----------
    inputs: Input tensor
    """
    variance_epsilon = 1e-5
    mean, variance = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
    outputs = (inputs - mean) / tf.sqrt(variance + variance_epsilon)
    return outputs


# ---------------
# util functions
# ---------------
def _make_pair(value):
    if isinstance(value, int):
        return (value,) * 2

    value_tuple = tuple(value)
    if len(value_tuple) != 2:
        raise ValueError(
            'value must be one interger or tuple of two intergers')
    return value_tuple
