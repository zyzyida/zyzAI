# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
import os

names = ["class", "text"]


def to_one_hot(y, n_class):
    return np.eye(n_class)[y.astype(int)]


def load_data(file_name, sample_ratio=1, n_class=2, names=names):
    '''load data from .csv file'''

    txt_file = pd.read_table(file_name, sep='\t', names=names)
    txt_file = txt_file.dropna(axis=0, how='any')
    shuffle_txt = txt_file  # .sample(frac=sample_ratio)
    #  pd.DataFrame.to_csv(shuffle_csv,"train.csv",sep='\t',header=False,index=False)

    x = pd.Series(shuffle_txt["text"])
    for i in x.index:
        # x[i] = x[i].decode('utf-8')
        x[i] = x[i].encode('utf-8')
    y = pd.Series(shuffle_txt["class"])
    y = to_one_hot(y, n_class)
    # print x[0:5]
    # for i in range(5):
    #    print x.values[i].decode('utf-8')
    # shuffle_csv = csv_file.sample(frac=sample_ratio)
    return x, y


def data_preprocessing(x_train, y_train, x_test, y_test, max_len, nums_train, nums_test):
    '''transform to one-hot idx vector by VocabularyProcess'''
    vocab_path = '../data/Vocabulary/vocab_processor'
    if os.path.exists(vocab_path):
        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    else:
        # 词汇处理器，max_len表示每一行文本最大的词数目，这里是20
        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_len, 10)
        vocab_processor.save(vocab_path)
    # fit_transform这里是做一个index,比如说第一行的文本就变成[1,2,3,4,5,...,0],一共20维
    x_transform_train = vocab_processor.fit_transform(x_train)
    # 这里的transform是在前面的基础上对测试集做index
    x_transform_test = vocab_processor.transform(x_test)
    # vocab是所有的词汇表
    vocab = vocab_processor.vocabulary_
    # 词汇表的大小，后面做one-hot的时候需要用
    vocab_size = len(vocab)
    x_train_list = list(x_transform_train)
    x_test_list = list(x_transform_test)
    x_train = np.array(x_train_list)
    x_test = np.array(x_test_list)

    return x_train[0:nums_train], y_train[0:nums_train], x_test[0:nums_test], y_test[0:nums_test], vocab, vocab_size


def takeVocab(x_train, x_test):
    print('x_train:', x_train)
    vocab_path = 'Vocabulary/vocab_processor'
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    # fit_transform这里是做一个index,比如说第一行的文本就变成[1,2,3,4,5,...,0],一共20维
    x_transform_train = vocab_processor.fit_transform(x_train)
    # 这里的transform是在前面的基础上对测试集做index
    x_transform_test = vocab_processor.transform(x_test)
    # vocab是所有的词汇表
    vocab = vocab_processor.vocabulary_
    print(vocab)
    # 词汇表的大小，后面做one-hot的时候需要用
    vocab_size = len(vocab)
    print('vocab_size', vocab_size)

    return vocab


# def data_train_preprocessing(x_train, y_train, x_test, y_test, max_len, nums_train, nums_test):
#     '''transform to one-hot idx vector by VocabularyProcess'''
#     vocab_path = 'Vocabulary/vocab_processor'
#     if os.path.exists(vocab_path):
#         vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(vocab_path)
#     else:
#         # 词汇处理器，max_len表示每一行文本最大的词数目，这里是200
#         vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_len)
#         vocab_processor.save(vocab_path)
#     # fit_transform这里是做一个index,比如说第一行的文本就变成[1,2,3,4,5,...,0],一共200维
#     x_transform_train = vocab_processor.fit_transform(x_train)
#     # 这里的transform是在前面的基础上对测试集做index
#     x_transform_test = vocab_processor.transform(x_test)
#     # vocab是所有的词汇表
#     vocab = vocab_processor.vocabulary_
#     # 词汇表的大小，后面做one-hot的时候需要用
#     vocab_size = len(vocab)
#     x_train_list = list(x_transform_train)
#     x_test_list = list(x_transform_test)
#     x_train = np.array(x_train_list)
#     x_test = np.array(x_test_list)
#
#     return x_train[0:nums_train], y_train[0:nums_train], x_test[0:nums_test], y_test[0:nums_test], vocab, vocab_size
#
#
# def data_test_preprocessing(x_train, y_train, x_test, y_test, max_len, nums_train, nums_test):
#     '''transform to one-hot idx vector by VocabularyProcess'''
#     vocab_path = 'Vocabulary/vocab_processor'
#     if os.path.exists(vocab_path):
#         vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(vocab_path)
#     else:
#         # 词汇处理器，max_len表示每一行文本最大的词数目，这里是200
#         vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_len)
#         vocab_processor.save(vocab_path)
#     # fit_transform这里是做一个index,比如说第一行的文本就变成[1,2,3,4,5,...,0],一共200维
#     x_transform_train = vocab_processor.fit_transform(x_train)
#     # 这里的transform是在前面的基础上对测试集做index
#     x_transform_test = vocab_processor.transform(x_test)
#     x_train_list = list(x_transform_train)
#     x_test_list = list(x_transform_test)
#     x_train = np.array(x_train_list)
#     x_test = np.array(x_test_list)
#
#     return x_train[0:nums_train], y_train[0:nums_train], x_test[0:nums_test], y_test[0:nums_test]


def split_dataset(x_test, y_test, dev_ratio):
    '''split test dataset to test and dev set with ratio '''
    test_size = len(x_test)
    print(test_size)
    dev_size = (int)(test_size * dev_ratio)
    print(dev_size)
    x_dev = x_test[:dev_size]
    x_test = x_test[dev_size:]
    y_dev = y_test[:dev_size]
    y_test = y_test[dev_size:]
    return x_test, x_dev, y_test, y_dev, dev_size, test_size - dev_size


def fill_feed_dict(data_X, data_Y, batch_size):
    '''Generator to yield batches'''
    # Shuffle data first.
    perm = np.random.permutation(data_X.shape[0])
    data_X = data_X[perm]
    data_Y = data_Y[perm]
    for idx in range(data_X.shape[0] // batch_size):
        x_batch = data_X[batch_size * idx: batch_size * (idx + 1)]
        y_batch = data_Y[batch_size * idx: batch_size * (idx + 1)]
        yield x_batch, y_batch
