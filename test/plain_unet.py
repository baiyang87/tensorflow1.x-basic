# -*- coding: utf-8 -*-
import tensorflow as tf
from unet.plain_unet import PlainUnet

if __name__ == '__main__':
    inputs = tf.ones(shape=(7, 256, 256, 3))
    net = PlainUnet()
    outputs = net.forward(inputs)
    print(outputs)
