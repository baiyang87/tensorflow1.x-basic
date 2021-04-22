# -*- coding: utf-8 -*-
"""
Basic layers implemented by keras
"""

from keras import backend as K
from keras.layers import Layer
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import ReLU
from keras.layers import LeakyReLU
from keras.layers import PReLU
from keras.layers import Softmax
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Multiply
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import Permute
from keras.layers import UpSampling2D


# ---------------
# basic layers
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
                  use_bias=use_bias)(inputs)


def dense(inputs, out_length, use_bias=True):
    """
    Dense connected layer

    Parameters
    ----------
    inputs: Input tensor
    out_length: Length of outputs
    use_bias: Whether to use bias
    """
    return Dense(units=out_length, use_bias=use_bias)(inputs)


def relu(inputs):
    """
    Relu activation

    Parameters
    ----------
    inputs: Input tensor
    """
    return ReLU()(inputs)


def leaky_relu(inputs, alpha=0.2):
    """
    Leaky relu activation

    Parameters
    ----------
    inputs: Input tensor
    alpha: Slope of negative neurons
    """
    return LeakyReLU(alpha=alpha)(inputs)


def prelu(inputs, shared_axes=(0, 1, 2, 3)):
    """
    Parametric relu activation

    Parameters
    ----------
    inputs: Input tensor
    shared_axes: The axes along which to share learnable parameters for the
        activation function.
    """
    return PReLU(shared_axes=shared_axes)(inputs)


def softmax(inputs, axis=-1):
    """
    Softmax activation

    Parameters
    ----------
    inputs: Input tensor
    axis: Axis along which the softmax normalization is applied
    """
    return Softmax(axis=axis)(inputs)


def batch_normalization(inputs, training=True):
    """
    Batch normalization

    Parameters
    ----------
    inputs: Input tensor
    training: Whether in training phase
    """
    return BatchNormalization()(inputs, training=training)


def maxpool(inputs, pool_size=(2, 2)):
    """
    Max pool

    Parameters
    ----------
    inputs: Input tensor
    pool_size: Size of the pooling window. One integer of tuple of two
        integers, (height_factor, width_factor)
    """
    return MaxPooling2D(pool_size=pool_size)(inputs)


def average_pool(inputs, pool_size=(2, 2)):
    """
    Average pool

    Parameters
    ----------
    inputs: Input tensor
    pool_size: Size of the pooling window. One integer of tuple of two
        integers, (height_factor, width_factor)
    """
    return AveragePooling2D(pool_size=pool_size)(inputs)


def global_average_pool(inputs):
    """
    Global average pool, the height and width dimension will be squeezed, that
    is, the dims of output tensor will be (batch, out_filters)

    Parameters
    ----------
    inputs: Input tensor
    """
    return GlobalAveragePooling2D()(inputs)


def concat(inputs_list):
    """
    Concatenate input tensors list

    Parameters
    ----------
    inputs_list: Input tensors list
    """
    return Concatenate()(inputs_list)


def add(inputs_list):
    """
    Add input tensors list

    Parameters
    ----------
    inputs_list: Input tensors list
    """
    return Add()(inputs_list)


def multiply(inputs_list):
    """
    Multiply input tensors list

    Parameters
    ----------
    inputs_list: Input tensors list
    """
    return Multiply()(inputs_list)


def dropout(inputs, drop_rate):
    """
    Applies Dropout to the input

    Parameters
    ----------
    inputs: Input tensor
    drop_rate: float between 0 and 1. Fraction of the input units to drop.
    """
    return Dropout(rate=drop_rate)(inputs)


def flatten(inputs):
    """
    Flattens the inputs

    Parameters
    ----------
    inputs: Input tensor
    """
    return Flatten()(inputs)


def reshape(inputs, new_shape):
    """
    Reshapes an output to a certain shape.

    Parameters
    ----------
    inputs: Input tensor
    new_shape: New shape without batch_size
    """
    return Reshape(target_shape=new_shape)(inputs)


def permute(inputs, new_dims):
    """
    Permutes the dimensions of the input according to a given pattern

    Parameters
    ----------
    inputs: Input tensor
    new_dims: Re-ordered dimensions
    """
    return Permute(dims=new_dims)(inputs)


def upsample(inputs, factor, interpolation='nearest'):
    """
    Upsampling layer by factor

    Parameters
    ----------
    inputs: Input tensor
    factor: The upsampling factors for (height, width). One integer or tuple of
        two integers
    interpolation: A string, one of [`nearest`, `bilinear`].
    """
    return UpSampling2D(size=factor, interpolation=interpolation)(inputs)


def instance_normalization(inputs):
    """
    Instance normalization layer

    Parameters
    ----------
    inputs: Input tensor
    """
    return InstanceNormalization()(inputs)


# ---------------
# combined layers
# ---------------
def conv_bn_relu(inputs, out_filters, ksize=(3, 3), strides=(1, 1),
                 dilation=(1, 1), use_bn=True, use_relu=True, training=True):
    """
    Combine conv, batch normalization (bn), relu. bn and relu are optional

    Parameters
    ----------
    inputs: Input tensor
    out_filters: Number of output filters
    ksize: Kernel size. One integer of tuple of two integers
    strides: Strides for moving kernel. One integer of tuple of two integers
    dilation: Dilation of kernel. One integer of tuple of two integers
    use_bn: Whether to use bn
    use_relu: Whether to use relu
    training: Whether in training phase
    """
    use_bias = not use_bn

    outputs = conv(inputs, out_filters, ksize, strides, dilation, use_bias)
    if use_bn:
        outputs = batch_normalization(outputs, training=training)
    if use_relu:
        outputs = relu(outputs)

    return outputs


def conv_in_relu(inputs, out_filters, ksize=(3, 3), strides=(1, 1),
                 dilation=(1, 1), use_in=True, use_relu=True):
    """
    Combine conv, instance normalization (in), relu. in and relu are optional

    Parameters
    ----------
    inputs: Input tensor
    out_filters: Number of output filters
    ksize: Kernel size. One integer of tuple of two integers
    strides: Strides for moving kernel. One integer of tuple of two integers
    dilation: Dilation of kernel. One integer of tuple of two integers
    use_in: Whether to use instance normalization
    use_relu: Whether to use relu
    """
    use_bias = not use_in

    outputs = conv(inputs, out_filters, ksize, strides, dilation, use_bias)
    if use_in:
        outputs = instance_normalization(outputs)
    if use_relu:
        outputs = relu(outputs)

    return outputs


def conv_in_lrelu(inputs, out_filters, ksize=(3, 3), strides=(1, 1),
                  dilation=(1, 1), use_in=True, use_lrelu=True, alpha=0.2):
    """
    Combine conv, instance normalization (in), leaky relu (lrelu).
    in and lrelu are optional

    Parameters
    ----------
    inputs: Input tensor
    out_filters: Number of output filters
    ksize: Kernel size. One integer of tuple of two integers
    strides: Strides for moving kernel. One integer of tuple of two integers
    dilation: Dilation of kernel. One integer of tuple of two integers
    use_in: Whether to use instance normalization
    use_lrelu: Whether to use leaky relu
    """
    use_bias = not use_in

    outputs = conv(inputs, out_filters, ksize, strides, dilation, use_bias)
    if use_in:
        outputs = instance_normalization(outputs)
    if use_lrelu:
        outputs = leaky_relu(outputs, alpha=alpha)

    return outputs


# ---------------
# auxiliaries
# ---------------
class InstanceNormalization(Layer):
    """
    Instance normalization layer
    """

    def __init__(self):
        super(InstanceNormalization, self).__init__()

    @classmethod
    def compute_output_shape(cls, input_shape):
        """
        Compute output shape.
        """
        return input_shape

    @classmethod
    def call(cls, inputs, variance_epsilon=1e-5):
        """
        Implementation of instance normalization

        Parameters
        ----------
        inputs: Input tensor
        variance_epsilon: epsilon added to variance to avoid zero divide
        """
        mean = K.mean(inputs, axis=[1, 2], keepdims=True)
        std = K.sqrt(
            K.var(inputs + variance_epsilon, axis=[1, 2], keepdims=True))
        outputs = (inputs - mean) / std
        return outputs
