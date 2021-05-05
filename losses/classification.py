# -*- coding: utf-8 -*-
import tensorflow as tf


def softmax_cross_entropy(onehot_labels, logits):
    """
    Cross entropy computed by softmax logits.

    Parameters
    ----------
    onehot_labels: One-hot-encoded labels
    logits: Logits outputs of the network.
    """
    onehot_labels = tf.cast(onehot_labels, logits.dtype)
    probabilities = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.reduce_sum(
        -onehot_labels * tf.log(probabilities), axis=-1))
    return loss


def sigmoid_cross_entropy(labels, logits):
    """
    Cross entropy computed by sigmoid logits.

    Parameters
    ----------
    labels: `[batch_size, num_classes]` target integer labels in `{0, 1}`
    logits: Float `[batch_size, num_classes]` logits outputs of the network
    """
    labels = tf.cast(labels, logits.dtype)
    zeros = tf.zeros_like(logits, dtype=logits.dtype)
    condition = tf.greater(logits, zeros)
    relu_logits = tf.where(condition, logits, zeros)
    loss = tf.reduce_mean(relu_logits - logits * labels +
                          tf.log1p(tf.exp(-tf.abs(logits))))
    return loss
