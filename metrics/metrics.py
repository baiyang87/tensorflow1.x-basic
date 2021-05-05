# -*- coding: utf-8 -*-
import tensorflow as tf


def accuracy(labels, predictions):
    """
    Compute accuracy that how often `predictions` matches `labels`.
    (refer to `tf.metrics.accuracy`)

    Parameters
    ----------
    labels: The ground truth values, a `Tensor` whose shape matches
        `predictions`
    predictions: The predicted values, a `Tensor` of any shape

    Returns
    -------
    acc: Accuracy, simply divides `total_num` by `correct_num`
    correct_num: Correct sample number
    total_num: Total sample number
    """
    is_correct = tf.equal(labels, predictions)

    correct_num = int(tf.reduce_sum(is_correct))
    total_num = int(predictions.get_shape()[0])
    acc = float(correct_num) / float(total_num)
    return acc, correct_num, total_num


def logits_accuracy(onehot_labels, logits, activation='softmax'):
    """
    Compute accuracy by logits.
    (refer to `tf.metrics.accuracy`)

    Parameters
    ----------
    onehot_labels: One-hot-encoded labels
    logits: Logits outputs of the network
    activation: Activation for logits, must be one of ['softmax', 'sigmoid']

    Returns
    -------
    acc: Accuracy, simply divides `total_num` by `correct_num`
    correct_num: Correct sample number
    total_num: Total sample number
    """
    if activation == 'softmax':
        onehot_labels = tf.cast(onehot_labels, logits.dtype)
        probabilities = tf.nn.softmax(logits)
        is_correct = tf.equal(tf.argmax(onehot_labels, 1),
                              tf.argmax(probabilities, 1))
    elif activation == 'sigmoid':
        ones = tf.ones_like(logits, dtype=logits.dtype)
        predictions = tf.greater_equal(tf.nn.sigmoid(logits), 0.5 * ones)
        labels = tf.greater_equal(onehot_labels, 0.5 * ones)
        is_correct = tf.equal(labels, predictions)
    else:
        raise ValueError("activation must be one of ['softmax', 'sigmoid']")

    correct_num = int(tf.reduce_sum(is_correct))
    total_num = int(logits.get_shape()[0])
    acc = float(correct_num) / float(total_num)
    return acc, correct_num, total_num
