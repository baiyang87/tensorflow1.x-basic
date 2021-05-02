# -*- coding: utf-8 -*-
import tensorflow as tf
from unet.plain_unet import PlainUnet

if __name__ == '__main__':
    inputs = tf.ones(shape=(7, 256, 256, 3))
    net = PlainUnet()
    net.config['output_channel'] = 20
    outputs = net.forward(inputs)
    print(outputs)
