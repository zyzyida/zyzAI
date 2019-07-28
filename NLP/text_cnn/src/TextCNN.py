# -*- coding: utf-8 -*-
# TextCNN: 1. embeddding layers, 2.convolutional layer, 3.max-pooling, 4.softmax layer.
import tensorflow as tf
import numpy as np
from numpy import *


class TextCNN:
    def __init__(self, filter_sizes, num_filters, num_classes, learning_rate, batch_size, decay_steps, decay_rate,
                 sequence_length, vocab_size, embed_size,
                 is_training, initializer=tf.random_normal_initializer(stddev=0.1), multi_label_flag=False,
                 clip_gradients=5.0, decay_rate_big=0.50):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")  # ADD learning_rate
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * decay_rate_big)
        self.filter_sizes = filter_sizes  # it is a list of int. e.g. [3,4,5]
        self.num_filters = num_filters
        self.initializer = initializer
        # self.num_filters_total = self.num_filters * len(filter_sizes)  # how many filters totally.
        self.num_filters_total = 3 * self.sequence_length - 3
        self.multi_label_flag = multi_label_flag
        self.clip_gradients = clip_gradients

        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        self.input_y = tf.placeholder(tf.int32, [None, ], name="input_y")  # y:[None,num_classes]
        self.input_y_multilabel = tf.placeholder(tf.float32, [None, self.num_classes],
                                                 name="input_y_multilabel")  # y:[None,num_classes]. this is for multi-label classification only.
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # self.iter = tf.placeholder(tf.int32, name="iter_name")  # training iteration
        # self.tst = tf.placeholder(tf.bool, [None, ], name="bn_name")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.b1 = tf.Variable(tf.ones([self.num_filters]) / 10)
        self.b2 = tf.Variable(tf.ones([self.num_filters]) / 10)
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits, self.alphas = self.inference()  # [None, self.label_size]. main computation graph is here.
        # self.logits = self.inference()  # [None, self.label_size]. main computation graph is here.
        self.possibility = tf.nn.sigmoid(self.logits)
        if not is_training:
            return
        if multi_label_flag:
            print("going to use multi label loss.")
            self.loss_val = self.loss_multilabel()
        else:
            print("going to use single label loss.")
            self.loss_val = self.loss()
        self.train_op = self.train()
        if not self.multi_label_flag:
            self.predictions = tf.argmax(self.logits, 1, name="predictions")  # shape:[None,]
            print("self.predictions:", self.predictions)
            correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32),
                                          self.input_y_multilabel)  # tf.argmax(self.logits, 1)-->[batch_size]
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"):  # embedding matrix
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer)  # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.W_projection = tf.get_variable("W_projection", shape=[self.num_filters_total, self.num_classes],
                                                initializer=self.initializer)  # [embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection",
                                                shape=[self.num_classes])  # [label_size] #ADD 2017.06.09

    def inference(self):
        """main computation graph here: 1.embedding-->2.CONV-BN-RELU-MAX_POOLING-->3.linear classifier"""
        # 1.=====>get emebedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)  # [None,sentence_length,embed_size]
        self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words,
                                                           -1)  # [None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
        # print(self.Embedding, self.input_x, self.embedded_words)
        print(self.sentence_embeddings_expanded)

        # attention_output, alphas_output = self.attention(self.sentence_embeddings_expanded)

        pooled_outputs = []
        attention_conv = []
        # h_pool = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution-pooling-%s" % filter_size):
                # ====>a.create filter
                print('filter_size:', filter_size)
                num_deep = self.sequence_length - filter_size + 1
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, num_deep],
                                         initializer=self.initializer)
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="VALID",
                                    name="coion-pooling-%s" % filter_size)  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                # conv = tf.nn.conv2d(attention_output, filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                print('conv:', conv)
                # conv, self.update_ema = self.batchnorm(conv, self.tst, self.iter, self.b1)

                # ====>c. apply nolinearity
                b = tf.get_variable("b-%s" % filter_size, [num_deep])
                h = tf.nn.relu(tf.nn.bias_add(conv, b),
                               "relu")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                print('h_layer:', h)

                self.h_attention = self.attention(h, filter_size, num_deep)
                print('self.h_attention:', self.h_attention)
                attention_conv.append(self.h_attention)
                # h_pool.append(h)

                pooled = tf.nn.avg_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID',
                                        name="pool")
                # pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                #                         strides=[1, 1, 1, 1], padding='VALID',
                #                         name="pool")

                pooled_outputs.append(pooled)

        print('attention_conv:', attention_conv)
        self.soft_3 = tf.concat(attention_conv, 1)
        print('self.soft_3:', self.soft_3)
        self.alphas = tf.nn.softmax(self.soft_3, dim=1)
        self.alphas = tf.squeeze(self.alphas, [2, 3])
        print('alphas_shape:', self.alphas)

        self.h_pool = tf.concat(pooled_outputs, 3)
        print('self.h_pool:', self.h_pool)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])

        self.h_pool_flat = self.h_pool_flat * self.alphas
        print('self.h_pool_flat:', self.h_pool_flat)

        # 4.=====>add dropout: use tf.nn.dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)
        self.h_drop = tf.layers.dense(self.h_drop, self.num_filters_total, activation=tf.nn.tanh, use_bias=True)

        # 5. logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            logits = tf.matmul(self.h_drop, self.W_projection) + self.b_projection
        return logits, self.alphas

    # 1-3gram
    def attention(self, inputs, filter_size, conv_deep):
        filter1 = tf.get_variable("attention-filter-%s" % filter_size, [1, 1, conv_deep, 128],
                                  initializer=self.initializer)

        conv_attention1 = tf.nn.conv2d(inputs, filter1, strides=[1, 1, 1, 1],
                                       padding="VALID", name="conv_attention1-%s" % filter_size)

        filter2 = tf.get_variable("attention-filter2-%s" % filter_size, [1, 1, 128, 1], initializer=self.initializer)
        conv_attention2 = tf.nn.conv2d(conv_attention1, filter2, strides=[1, 1, 1, 1], padding="VALID",
                                       name="conv_attention2-%s" % filter_size)
        # 加relu


        # alphas = tf.nn.softmax(conv_attention2, dim=1)
        # print('alphas:', alphas.shape)
        #
        # output = inputs * alphas
        # # output = tf.matmul(inputs, alphas)
        # print('output:', output.shape)

        return conv_attention2

    def batchnorm(self, Ylogits, is_test, iteration, offset,
                  convolutional=False):
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_averages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_averages

    def loss_multilabel(self, l2_lambda=0.0001):  # 0.0001#this loss function is for multi-label classification
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,
                                                             logits=self.logits)
            print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999).
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            loss = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def loss(self, l2_lambda=0.0001):  # 0.001
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,
                                                                    logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer="Adam",
                                                   clip_gradients=self.clip_gradients)
        return train_op

    # def attention(self, inputs, attention_size, time_major=True, return_alphas=False):
    #     print("inputs shape={}".format(inputs.shape)) # input=（?,200,128,1）
    #
    #     hidden_size = inputs.shape[1].value  # 128 隐藏层神经元个数
    #     # hidden_size = (inputs.shape[1].value) * (inputs.shape[2].value) * (inputs.shape[3].value)
    #     # Trainable parameters
    #     W_omega = tf.get_variable("W_omega", [hidden_size, attention_size], dtype=tf.float32)
    #     b_omega = tf.get_variable("b_omega", [attention_size], dtype=tf.float32)
    #     u_omega = tf.get_variable("u_omega", [attention_size, 1], dtype=tf.float32)
    #
    #     # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    #     # the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
    #     inputs_re = tf.reshape(inputs, [-1, hidden_size])  # 30*128，256
    #     v = tf.tanh(tf.matmul(inputs_re, W_omega) + b_omega)  # 30*128 ，128
    #     print(inputs_re, v, u_omega)
    #     # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    #     vu = tf.matmul(v, u_omega)  # (B,T) shape  30*128 1
    #     print(vu)
    #     vu = tf.reshape(vu, [-1, 128])  # 128 30
    #     print(vu)
    #     alphas = tf.nn.softmax(vu)  # (B,T) shape also
    #     print('alphas:', alphas)
    #
    #     # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    #     output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)  # tf.reduce_sum()按行求和
    #     # inputs.shape=128 30 128 alphas.shape=128,30,1 output.shape=128,30,128
    #     # output = tf.reshape(output, (-1, 200, 128, 1))
    #
    #     return output

    # 1-gram
    # def attention(self, inputs):
    #     # print("inputs shape={}".format(inputs.shape))  # input=（?,200,128,1）
    #
    #     filter1 = tf.get_variable("attention-filter-%s" % 1, [1, self.embed_size, 1, 64], initializer=self.initializer)
    #     # print('filter1:', filter1)
    #     conv_attention1 = tf.nn.conv2d(self.sentence_embeddings_expanded, filter1, strides=[1, 1, 1, 1],
    #                                    padding="VALID", name="conv_attention1")
    #     # print('conv_attention1:', conv_attention1)
    #
    #     filter2 = tf.get_variable("attention-filter2-%s" % 1, [1, 1, 64, 1], initializer=self.initializer)
    #     conv_attention2 = tf.nn.conv2d(conv_attention1, filter2, strides=[1, 1, 1, 1], padding="VALID",
    #                                    name="conv_attention2")
    #     # print('conv_attention2:', conv_attention2)
    #
    #     alphas = tf.nn.softmax(conv_attention2, dim=1)
    #     print('alphas:', alphas.shape)
    #
    #     output = inputs * alphas
    #     # output = tf.matmul(inputs, alphas)
    #     print('output:', output.shape)
    #
    #     return output, alphas


def test():
    num_classes = 5
    learning_rate = 0.001
    batch_size = 8
    decay_steps = 1000
    decay_rate = 0.95
    sequence_length = 5
    vocab_size = 10000
    embed_size = 100
    is_training = True
    dropout_keep_prob = 1.0  # 0.5
    filter_sizes = [2, 3, 4]
    num_filters = 128
    multi_label_flag = True
    textRNN = TextCNN(filter_sizes, num_filters, num_classes, learning_rate, batch_size, decay_steps, decay_rate,
                      sequence_length, vocab_size, embed_size, is_training, multi_label_flag=multi_label_flag)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(500):
            input_x = np.random.randn(batch_size, sequence_length)  # [None, self.sequence_length]
            input_x[input_x >= 0] = 1
            input_x[input_x < 0] = 0
            input_y_multilabel = get_label_y(input_x)
            loss, possibility, W_projection_value, _ = sess.run(
                [textRNN.loss_val, textRNN.possibility, textRNN.W_projection, textRNN.train_op],
                feed_dict={textRNN.input_x: input_x, textRNN.input_y_multilabel: input_y_multilabel,
                           textRNN.dropout_keep_prob: dropout_keep_prob})
            print(i, "loss:", loss, "-------------------------------------------------------")
            print("label:", input_y_multilabel);
            print("possibility:", possibility)


def get_label_y(input_x):
    length = input_x.shape[0]
    input_y = np.zeros((input_x.shape))
    for i in range(length):
        element = input_x[i, :]  # [5,]
        result = compute_single_label(element)
        input_y[i, :] = result
    return input_y


def compute_single_label(listt):
    result = []
    length = len(listt)
    for i, e in enumerate(listt):
        previous = listt[i - 1] if i > 0 else 0
        current = listt[i]
        next = listt[i + 1] if i < length - 1 else 0
        summ = previous + current + next
        if summ >= 2:
            summ = 1
        else:
            summ = 0
        result.append(summ)
    return result
