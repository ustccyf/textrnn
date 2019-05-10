#using coding=utf8
import sys,os
from config import Config
import tensorflow as tf
import numpy as np
import logging

class rnn():
    def __init__(self, config):
        self.config = config
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'gpu': 0}))
        self.input_x = tf.placeholder(tf.int32, [None, None], name = "input_x")
        self.input_y = tf.placeholder(tf.int32, [None], name = "input_y")
        self.sequence_lengths = tf.placeholder(tf.int32, shape = [None], name = "sequence_lengths" )
        self.dropout = tf.placeholder(tf.float32, name='keep_prob')
        self.num_classes = config.num_classes
        self.embedding_dim = config.embedding_dim
        self.seq_length = config.seq_length
        self.vocab_size = config.vocab_size
        self.hidden_size_lstm = 64
        self.attention_size = 50
        self.mlp_num_units = 64
        self.lr_method = "adam"
        self.clip = -1
        self.lr = config.learning_rate
    def build(self):
        #word embedding layer
        with tf.variable_scope("emb_layer"):
            if self.config.embeddings is None:
                embed_matrix = tf.get_variable(
                        name = "_word_embeddings",
                        dtype = tf.float32,
                        shape=[self.vocab_size, self.embedding_dim])
                self.embedding_trainable = True
            else:
                embed_matrix = tf.Variable(
                        self.config.embeddings,
                        name = "_word_embeddings",
                        dtype = ft.float32,
                        trainable = False)
                self.embedding_trainable = False
            word_embbeddings_lookup = tf.nn.embedding_lookup(embed_matrix,
                    self.input_x, name = "word_embedding");
            self.word_embeddings = tf.nn.dropout(word_embbeddings_lookup,
                    self.dropout, name="word_embeddings")
        #rnn layer
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length = self.sequence_lengths,
                    dtype=tf.float32, scope="bi-lstm")
            output = tf.concat([output_fw, output_bw], axis=-1,
                    name = "lstm_concat")
            self.lstm_outputs = tf.nn.dropout(output, self.dropout,
                    name = "lstm_output")
        #attention layer
        with tf.variable_scope("attention"):
            ##学习一个向量v_att，对lstm输出点乘之后softmax之后对所有lstm输出进行加权,得到最终的query的表示
            #nsteps = tf.shape(self.lstm_outputs, name = "att_nsteps")[1]
            #sequence(每句话)的长度
            max_time = tf.shape(self.lstm_outputs, name = "max_time")[1]
            atten_size = self.attention_size
            lstm_output = tf.reshape(self.lstm_outputs,[-1, 2 * self.hidden_size_lstm], name = "output_att")
            W_omega = tf.Variable(tf.random_normal(\
                [2 * self.hidden_size_lstm, atten_size], stddev = 0.1, 
                dtype = tf.float32), name = "w_omega")
            b_omega = tf.Variable(tf.random_normal(\
                [atten_size], stddev = 0.1,\
                dtype = tf.float32), name= "b_omega")
            u_omega = tf.Variable(tf.random_normal(\
                    [atten_size], stddev = 0.1, dtype = tf.float32),\
                    name = "u_omega")
            pred = tf.matmul(lstm_output, W_omega) + tf.reshape(b_omega, [1, -1])
            v = tf.tanh(pred, name = "v")
            vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
            exps = tf.reshape(tf.exp(vu), [-1, max_time], name="vu")
            #sequese每个位置对应一个exps的权重
            alphas = exps/ tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
            atten_outs = tf.reduce_sum( \
                    self.lstm_outputs * tf.reshape(alphas, [-1, max_time, 1]), 1)
            self.attention_out = atten_outs
        #mlp layer
        with tf.variable_scope("classification_mlp"):
            self.mlp_output = tf.contrib.layers.fully_connected(
                    inputs=self.attention_out,
                    num_outputs = self.mlp_num_units,
                    activation_fn=tf.nn.relu,
                    biases_initializer=tf.random_uniform_initializer(),
                    weights_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                    scope="classification_mlp")
        #logit layer
        with tf.variable_scope("logits"):
            self.logits = tf.contrib.layers.fully_connected(
                    inputs=self.mlp_output,
                    num_outputs=self.num_classes,
                    activation_fn=None,
                    weights_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                    biases_initializer=tf.random_uniform_initializer(),
                    scope="intent_logits")
            self.y = tf.argmax(self.logits, 1)
            self.score = tf.nn.softmax(self.logits)
        with tf.variable_scope("loss_op"):
            label = tf.one_hot(self.input_y, self.config.num_classes,
                    on_value=1, off_value=0)
            losses = tf.nn.softmax_cross_entropy_with_logits(
                    labels=label,logits=self.logits)
            self.loss = tf.reduce_mean(losses)
            tf.summary.scalar("loss", self.loss)
    def train(self):
        _lr_m = self.lr_method.lower()
        if _lr_m == 'adam':
            optimizer = tf.train.AdamOptimizer(lr, name="optimizer")
        elif _lr_m == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(lr, name="optimizer")
        elif _lr_m == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(lr, name="optimizer")
        else:
            optimizer = tf.train.AdamOptimizer(lr)
        if clip > 0:
            grads, vs     = zip(*optimizer.compute_gradients(loss))
            grads, gnorm  = tf.clip_by_global_norm(grads, clip)
            self.train_op = optimizer.apply_gradients(zip(grads, vs))
        else:
            self.train_op = optimizer.minimize(loss)
    def initialize_session(self):
        self.logger.info("Initializing tf session")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

if __name__ == "__main__":
    config = Config()
    rnn_model = rnn(config)
    rnn_model.build()    


