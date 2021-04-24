# -*- coding: utf-8 -*-
import tensorflow as tf
from classification_net.res_net import ResNet

if __name__ == '__main__':
    inputs = tf.ones(shape=(7, 256, 256, 3))
    net = ResNet()
    net.config['num_classes'] = 20
    outputs = net.forward(inputs)
    print(outputs)
