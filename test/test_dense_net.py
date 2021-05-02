# -*- coding: utf-8 -*-
import tensorflow as tf
from classification_net.dense_net import DenseNet

if __name__ == '__main__':
    inputs = tf.ones(shape=(7, 256, 256, 3))
    net = DenseNet()
    net.config['num_classes'] = 20
    outputs = net.forward(inputs)
    print(outputs)
