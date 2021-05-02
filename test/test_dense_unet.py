# -*- coding: utf-8 -*-
import tensorflow as tf
from unet.dense_unet import DenseUnet

if __name__ == '__main__':
    inputs = tf.ones(shape=(7, 256, 256, 3))
    net = DenseUnet()
    net.config['output_channel'] = 20
    outputs = net.forward(inputs)
    print(outputs)
