import numpy as np
import tensorflow as tf


def gcn_layer(x,
              adj,
              scope='GCN',
              size=64,
              activation=tf.nn.relu,
              dropout=1.,
              training=False):
    """
    inputs: tensor of node features (or hidden features)
    adj: sparse tensor of normalized adjacent matrix (output of utils.preprocess_adj())
    scope: name scope of the layer
    activation: activation function of the GCN layer
    dropout: dropout keep probability, default to 1. (no dropout)
    training: training mode, whether use dropout

    """
    with tf.variable_scope(scope):
        net = tf.sparse_tensor_dense_matmul(adj, x)
        net = tf.contrib.layers.fully_connected(inputs=net,
                                                num_outputs=size,
                                                activation_fn=activation)
        net = tf.contrib.layers.dropout(inputs=net,
                                        keep_prob=dropout,
                                        is_training=training)
    return net


def fc_dropout(x,
               scope="fc_dropout",
               size=None,
               dropout=1.,
               activation=tf.nn.elu,
               training=True):
    assert size, "Must specify layer size (num nodes)"
    # use linear activation for pre-activation batch_normalization

    with tf.variable_scope(scope):
        fc = tf.contrib.layers.fully_connected(inputs=x,
                                               num_outputs=size,
                                               activation_fn=activation)

        fc_drop = tf.contrib.layers.dropout(inputs=fc,
                                            keep_prob=dropout,
                                            is_training=training)

    return fc_drop


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)