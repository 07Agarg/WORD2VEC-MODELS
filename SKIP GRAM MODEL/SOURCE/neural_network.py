# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 07:06:16 2018

@author: ashima.garg
"""

import tensorflow as tf
import config

class Embedding_Layer():
    
    def __init__(self, shape):
        self.embedding = tf.get_variable("embedding", shape=shape, dtype=tf.float32)
        
    def lookup(self, input_data):
        output = tf.nn.embedding_lookup(self.embedding, input_data)
        return output


class NCE_Layer():
    
    def __init__(self, shape):
        self.weights = tf.Variable(tf.truncated_normal([config.VOCABULARY_SIZE, config.EMBEDDING_SIZE]), stddev = 1.0/math.sqrt(shape[1]))
        self.bias = tf.Variable(tf.zeros(shape = shape[0]))
        
    def loss(self, input_data, labels):
        loss = tf.reduce_mean(tf.nn.nce_loss(self.weights, self.biases, labels, input_data, num_sampled = config.NUM_SAMPLED, num_classes = config.VOCABULARY_SIZE))
        return loss