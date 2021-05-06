# -*- coding: utf-8 -*-
from datasets.mnist import load as load_mnist
from losses.classification import softmax_cross_entropy
from losses.classification import sigmoid_cross_entropy

from losses.regression import mean_absolute_error
from losses.regression import mean_squared_error
from losses.regression import huber_loss
