# -*- coding: utf-8 -*-
import tensorflow as tf


def mean_absolute_error(labels, predictions):
    """
    Mean Absolute Error, or L1 loss.
    (refer to `tf.losses.absolute_difference`)

    Parameters
    ----------
    labels: The ground truth output tensor, same dimensions as 'predictions'
    predictions: The predicted output tensor
    """
    return tf.reduce_mean(tf.abs(labels - predictions))


def mean_squared_error(labels, predictions):
    """
    Mean Squared Error, or L2 loss.
    (refer to `tf.losses.mean_squared_error`)

    Parameters
    ----------
    labels: The ground truth output tensor, same dimensions as 'predictions'
    predictions: The predicted output tensor
    """
    return tf.reduce_mean(tf.square(labels - predictions))


def huber_loss(labels, predictions, delta=1.0):
    """
    Huber loss.
    (refer to `tf.losses.huber_loss`)

    For each value x in `error=labels-predictions`, the following is
    calculated:
    ```
    0.5 * x^2                  if |x| <= d
    0.5 * d^2 + d * (|x| - d)  if |x| > d
    ```
    where d is `delta`

    Parameters
    ----------
    labels: The ground truth output tensor, same dimensions as 'predictions'
    predictions: The predicted output tensor
    delta: `float`, the point where the huber loss function changes from a
        quadratic to linear.
    """
    abs_error = tf.abs(labels - predictions)
    condition = tf.less(abs_error, delta)
    quadratic = 0.5 * tf.square(abs_error)
    linear = delta * abs_error - 0.5 * tf.square(delta)
    return tf.reduce_mean(tf.where(condition, quadratic, linear))
