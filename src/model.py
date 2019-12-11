#coding: utf-8

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import time

def prelu(x, name=None):
    with tf.variable_scope(name,'prelu'):
        i = int(x.get_shape()[-1])
        alpha = tf.get_variable('alpha',
                                shape=(i,),
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0.25))
        y = tf.nn.relu(x) + tf.multiply(alpha, -tf.nn.relu(-x))
    return y

class CausalConv1d(tf.layers.Conv1D):
    def __init__(self,
            filters,
            kernel_size,
            strides=1,
            dilation_rate=1,
            activation=None,
            use_bias=True,
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            name=None,
            **kwargs):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name, **kwargs
            )

    def call(self, x):
        padding = (self.kernel_size[0] - 1) * self.dilation_rate[0]
        x = tf.pad(x, tf.constant([[0,0],[padding,0],[0,0]],dtype=tf.int32))
        return super().call(x)

class TemporalBlock(tf.layers.Layer):
    def __init__(self, n_outputs, kernel_size, strides, dilation_rate, dropout=0.2,
        trainable=True, name=None, dtype=None, activity_regularizer=None, **kwargs):
        super().__init__(trainable=trainable, dtype=dtype, activity_regularizer=activity_regularizer,
            name=name, **kwargs)
        
        self._n_outputs = n_outputs
        self._kernel_size = kernel_size
        self._strides = strides
        self._dilation_rate = dilation_rate
        self._dropout = dropout

    def build(self, input_shape):
        self._conv1 = CausalConv1d(self._n_outputs, self._kernel_size, strides=self._strides,
            dilation_rate=self._dilation_rate, activation=None, name='conv1')
        self._conv2 = CausalConv1d(self._n_outputs, self._kernel_size, strides=self._strides,
            dilation_rate=self._dilation_rate, activation=None, name='conv2')

        self._dropout1 = tf.layers.Dropout(self._dropout, [tf.constant(1), tf.constant(1), tf.constant(self._n_outputs)])
        self._dropout2 = tf.layers.Dropout(self._dropout, [tf.constant(1), tf.constant(1), tf.constant(self._n_outputs)])

        if input_shape[2] != self._n_outputs:
            #self.down_sample = tf.layers.Conv1D(
            #     self.n_outputs, kernel_size=1, 
            #     activation=None, data_format="channels_last", padding="valid")
            self._down_sample = tf.layers.Dense(self._n_outputs, activation=None)
        else:
            self._down_sample = None

    def call(self, x, training=True):
        y = self._conv1(x)
        y = tf.contrib.layers.layer_norm(y)
        y = prelu(y, name = self.name+'conv1_prelu')
        y = self._dropout1(y, training=training)

        y = self._conv2(y)
        y = self._dropout2(y, training=training)

        if self._down_sample is not None:
            x = self._down_sample(x)
        y = tf.contrib.layers.layer_norm(x+y)
        return prelu(y, name=self.name+'conv2_prelu')

class TemporalConvNet(tf.layers.Layer):
    def __init__(self, num_channels, kernel_size=2, dropout=0.2,
        trainable=True, name=None, dtype=None, activity_regularizer=None, **kwargs):
        super().__init__(trainable=trainable, dtype=dtype, activity_regularizer=activity_regularizer,
            name=name, **kwargs)
        self._layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            self._layers.append(TemporalBlock(out_channels, kernel_size, strides=1, 
                dilation_rate=dilation_size, dropout=dropout, name='tblock_{}'.format(i)))

    def call(self, x, training=True):
        y = x
        for layer in self._layers:
            y = layer(y, training=training)
        return y

class SENetLayer(tf.layers.Layer):
    def __init__(self, out_dim, ratio, trainable=True, name=None, dtype=None, activity_regularizer=None, **kwargs):
        super().__init__(trainable=trainable, dtype=dtype, activity_regularizer=activity_regularizer, 
            name=name, **kwargs)
        self._out_dim = out_dim
        self._ratio = ratio


    def build(self, input_shape):
        self._squeeze = tf.layers.AveragePooling1D((input_shape[1],), 1, name='squeeze')
        self._excitation_1 = tf.layers.Dense(self._out_dim // self._ratio, activation=prelu, name='excitation_1')
        self._excitation_2 = tf.layers.Dense(self._out_dim, activation=tf.nn.sigmoid, name='excitation_2')

    def call(self, x):
        squeeze = self._squeeze(x)
        excitation = self._excitation_1(squeeze)
        excitation = self._excitation_2(excitation)
        excitation = tf.reshape(excitation, [-1, 1, self._out_dim])
        scale = x * excitation
        return scale

class Network(object):
    """docstring for Network"""
    def __init__(self, tcn_num_channels, tcn_kernel_size, tcn_dropout, embedding_size, weight_decay, num_classes):

        self._tcn_num_channels = tcn_num_channels
        self._tcn_kernel_size = tcn_kernel_size
        self._tcn_dropout = tcn_dropout
        self._embedding_size = embedding_size
        self._weight_decay = weight_decay
        self._num_classes = num_classes

    def __build__(self, input_shape):
        self._tcn = TemporalConvNet(self._tcn_num_channels, self._tcn_kernel_size, 
            self._tcn_dropout, name='TCN')
        self._se_layer = SENetLayer(self._tcn_num_channels[-1], 4, name='SENetLayer')

        #self._embedding_layer = tf.layers.Dense(self._embedding_size, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(),
        #   kernel_regularizer=tf.contrib.layers.l2_regularizer(self._weight_decay), name='bottleneck')
        self._softmax_layer = tf.layers.Dense(self._num_classes, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_regularizer=None, name='logits')

    def __call__(self, x, training=True):
        self.__build__(x.shape)

        y = self._tcn(x, training=training)
        y = self._se_layer(y)
        prelogits = tf.squeeze(tf.layers.average_pooling1d(y, y.shape.as_list()[1], strides=1), axis=1, name='prelogits')
        #prelogits = self._embedding_layer(y[:, -1, :])
        #prelogits = y[:, -1, :]
        logits = self._softmax_layer(prelogits)
        embeddings = tf.nn.l2_normalize(prelogits, axis=1, name='embeddings')
        return prelogits, logits, embeddings
             
