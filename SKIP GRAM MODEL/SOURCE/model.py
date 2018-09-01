# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 07:06:16 2018

@author: ashima.garg
"""


import tensorflow as tf
import config
import neural_network
import os

class MODEL():
        def __init__(self):
            self.inputs = tf.placeholder(shape=[config.BATCH_SIZE], dtype=tf.int32)
            self.labels = tf.placeholder(shape=[config.BATCH_SIZE, 1], dtype=tf.float32)
            self.embeddings = None
            self.loss = None

        def build(self):
            e_layer = neural_network.Embedding_Layer([config.VOCABULARY_SIZE, config.EMBEDDING_SIZE])
            embeddings = e_layer.lookup(self.inputs)
            self.embeddings = e_layer.embedding
            
            nce_layer = neural_network.NCE_Layer([config.VOCABULARY_SIZE, config.EMBEDDING_SIZE])
            self.loss = nce_layer.loss(embeddings, self.labels)

        def train(self, data):
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
            normalized_embeddings = self.embeddings / norm
            saver = tf.train.Saver()
            with tf.Session() as session:
                init = tf.global_variables_initializer()
                session.run(init)
                print("Variables initialized...")
                average_loss = 0
                for step in range(config.NUM_STEPS):
                    batch_X, batch_Y = data.generate_batch()
                    feed_dict = {self.inputs: batch_X, self.labels: batch_Y}
                    _, loss_val = session.run([optimizer, self.loss], feed_dict=feed_dict)
                    average_loss += loss_val
                    if step % 2000 == 0:
                        if step > 0:
                            average_loss = average_loss / 2000
                        print("Average loss at step: ", step, " average_loss: ", average_loss)
                        average_loss = 0
                self.embeddings = normalized_embeddings.eval()
                save_path = saver.save(session, os.path.join(config.MODEL_DIR, "model" + str(config.NUM_SKIPS) + str(config.SKIP_WINDOW)))
                print("Model saved in path: %s" % save_path)
                