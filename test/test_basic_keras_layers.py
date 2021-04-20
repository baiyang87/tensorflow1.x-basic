# -*- coding: utf-8 -*-
import keras

from basic_layers.keras import *


def net(inputs):
    outputs = inputs
    outputs = conv(outputs, 33)
    outputs = relu(outputs)
    outputs = leaky_relu(outputs, 0.2)
    outputs = prelu(outputs)
    outputs = prelu(outputs, (1, 2, 3))
    outputs = softmax(outputs)
    outputs = batch_normalization(outputs)
    outputs = instance_normalization(outputs)
    outputs = concat([outputs, outputs])
    outputs = add([outputs, outputs])
    outputs = multiply([outputs, outputs])
    outputs = reshape(outputs, (128, 128, -1))
    outputs = upsample(outputs, 2, 'bilinear')
    outputs = maxpool(outputs)
    outputs = average_pool(outputs)
    outputs = global_average_pool(outputs)
    return outputs


if __name__ == '__main__':
    inputs = keras.Input((256, 256, 3))
    outputs = net(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary(line_length=150))
