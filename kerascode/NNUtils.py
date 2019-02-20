#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'lkjie'

import os
import keras
from keras import backend as K
from keras.layers import Lambda
import tensorflow as tf
from tensorflow.python.ops import array_ops


def OneHot(input_dim=None, input_length=None):
    # Check if inputs were supplied correctly
    if input_dim is None or input_length is None:
        raise TypeError("input_dim or input_length is not set")

    # Helper method (not inlined for clarity)
    def _one_hot(x, num_classes):
        return K.one_hot(K.cast(x, 'uint8'),
                         num_classes=num_classes)

    # Final layer representation as a Lambda layer
    return Lambda(_one_hot,
                  arguments={'num_classes': input_dim},
                  input_shape=(input_length,))


def top1(y_true, y_pred):
    return keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, 1)


def top3(y_true, y_pred):
    return keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, 3)


def top5(y_true, y_pred):
    return keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, 5)


def top10(y_true, y_pred):
    return keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, 10)


#
# def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
#     r"""Compute focal loss for predictions.
#         Multi-labels Focal loss formula:
#             FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
#                  ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
#     Args:
#      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
#         num_classes] representing the predicted logits for each class
#      target_tensor: A float tensor of shape [batch_size, num_anchors,
#         num_classes] representing one-hot encoded classification targets
#      weights: A float tensor of shape [batch_size, num_anchors]
#      alpha: A scalar tensor for focal loss alpha hyper-parameter
#      gamma: A scalar tensor for focal loss gamma hyper-parameter
#     Returns:
#         loss: A (scalar) tensor representing the value of the loss function
#     """
#     sigmoid_p = tf.nn.sigmoid(prediction_tensor)
#     zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
#
#     # For positive prediction, only need consider front part loss, back part is 0;
#     # target_tensor > zeros <=> z=1, so positive coefficient = z - p.
#     pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
#
#     # For negative prediction, only need consider back part loss, front part is 0;
#     # target_tensor > zeros <=> z=1, so negative coefficient = 0.
#     neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
#     per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
#                           - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
#     return tf.reduce_sum(per_entry_cross_ent)


def focal_loss(prediction_tensor, target_tensor, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    # sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

    # For positive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so positive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - prediction_tensor, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, prediction_tensor)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(
        tf.clip_by_value(1.0 - prediction_tensor, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)


def focal_loss_noalpha(prediction_tensor, target_tensor, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    # sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

    # For positive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so positive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - prediction_tensor, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, prediction_tensor)
    per_entry_cross_ent = - (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0)) \
                          - (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - prediction_tensor, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)


def focal_loss_noalphav2(prediction_tensor, target_tensor, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    # sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

    # For positive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so positive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - prediction_tensor, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, prediction_tensor)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0)) \
                          - (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - prediction_tensor, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)


def focal_loss_with_sigmoid(prediction_tensor, target_tensor, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For positive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so positive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)


def focal_loss_pos(prediction_tensor, target_tensor, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    # sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

    # For positive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so positive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - prediction_tensor, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, prediction_tensor)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)


def mertics(y_true, y_pred):
    def topk_tf(ytrue, ypred, k=1):
        assert len(ytrue.shape) == 1 and len(ypred.shape) == 2
        predictions = tf.Variable(ypred, dtype=tf.float32)
        targets = tf.Variable(ytrue)
        topk = tf.nn.in_top_k(
            predictions,
            targets,
            k,
            name=None
        )
        tfsess.run(predictions.initializer)
        tfsess.run(targets.initializer)
        res = tfsess.run(topk)
        return res.sum() / res.shape[0]

    init_op = tf.global_variables_initializer()
    config = tf.ConfigProto(device_count={"CPU": 1, 'GPU': 0})
    # tfsess = tf.InteractiveSession(config=config)
    tfsess = tf.Session(config=config)
    tfsess.run(init_op)
    top1 = topk_tf(y_true, y_pred, k=1)
    top3 = topk_tf(y_true, y_pred, k=3)
    top5 = topk_tf(y_true, y_pred, k=5)
    top10 = topk_tf(y_true, y_pred, k=10)
    print('top1, top3, top5, top10')
    print(top1, top3, top5, top10)
