# encoding=utf8
import os
import time
from prepareData import *
import tensorflow as tf
from TextCNN import TextCNN
import numpy as np
import sys

# reload(sys)
# sys.setdefaultencoding('utf8')

# gpu设置
# config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
# config.gpu_options.allow_growth = True
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# configuration
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("vocab_size", 548808, "maximum vocab size.")
# tf.app.flags.DEFINE_integer("vocab_size", 51987, "maximum vocab size.")  # 词表大小
tf.app.flags.DEFINE_float("learning_rate", 0.003, "learning rate")  # 学习率
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size for training/evaluating.")  # 批处理的大小
tf.app.flags.DEFINE_integer("decay_steps", 10000, "how many steps before decay learning rate.")  # learning rate多少次衰减一次
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.")  # 一次衰减多少
tf.app.flags.DEFINE_integer("sentence_len", 20, "max sentence length")  # 句子的词长度
tf.app.flags.DEFINE_integer("embed_size", 128, "embedding size")  # 词向量大小
tf.app.flags.DEFINE_boolean("is_training", True, "is traning.true:tranining,false:testing/inference")  # 是否训练
tf.app.flags.DEFINE_integer("num_epochs", 5, "number of epochs to run.")  # 迭代轮次
tf.app.flags.DEFINE_integer("num_filters", 128, "number of filters")  # 卷积大小
tf.app.flags.DEFINE_boolean("multi_label_flag", True, "use multi label or single label.")  # 是否为多类别
tf.app.flags.DEFINE_integer("label_number", 2, "number of labels.")  # 类别数量
filter_sizes = [1, 2, 3]  # 卷积核大小


# 加载数据
def get_train_and_test_data():
    print('load test data......')
    x_test, y_test = load_data("../data/data_all/dataall_test.txt")
    print('load train data......')
    x_train, y_train = load_data("../data/data_all/dataall_train.txt")

    nums_train = len(x_train)
    nums_test = 10000
    x_train, y_train, x_test, y_test, vocab, vocab_size = data_preprocessing(x_train, y_train, x_test, y_test,
                                                                             FLAGS.sentence_len, nums_train=nums_train,
                                                                             nums_test=nums_test)

    x_test, x_dev, y_test, y_dev, dev_size, test_size = split_dataset(x_test, y_test, 0.1)

    print('vocab_size:', vocab_size)
    # 保存数据为npy格式，读取速度快
    np.save('../data/cnn_data/x_train.npy', x_train)
    np.save('../data/cnn_data/y_train.npy', y_train)
    np.save('../data/cnn_data/x_dev.npy', x_dev)
    np.save('../data/cnn_data/y_dev.npy', y_dev)
    np.save('../data/cnn_data/x_test.npy', x_test)
    np.save('../data/cnn_data/y_test.npy', y_test)
    np.save('../data/cnn_data/vocab.pickle', vocab)

    return x_train, y_train, x_dev, y_dev, x_test, y_test, vocab


# 读取npy数据文件
def get_data():
    x_train = np.load('..data/cnn_data/x_train.npy')
    y_train = np.load('../data/cnn_data/y_train.npy')
    x_dev = np.load('../data/cnn_data/x_dev.npy')
    y_dev = np.load('../data/cnn_data/y_dev.npy')
    x_test = np.load('../data/cnn_data/x_test.npy')
    y_test = np.load('../data/cnn_data/y_test.npy')
    vocab = np.load('../data/cnn_data/vocab.pickle')
    return x_train, y_train, x_dev, y_dev, x_test, y_test, vocab


# 结果存入txt中
def train_write(alpha, x_batch, y_batch, vocab, f):
    for line in range(FLAGS.batch_size):
        label = y_batch[line, 0]
        # print('label:', label)  # 0:非色情，1：色情；（和原始label相反）
        if label == 1:
            x_title = x_batch[line, :]
            att = alpha[line, :]
            att_top = np.argsort(att)
            for j in x_title:
                if j != 0:
                    # print(vocab.reverse(j))
                    f.write(vocab.reverse(j))
            f.write('   ')
            index = []
            number = -5
            for i in range(number, 0):
                index.append(att_top[i])
            for i in range(number, 0):
                # print(att[att_top[i]])
                f.write(str(att[att_top[i]]) + ' ')
            x_title = x_title.tolist()
            # print('----result----')
            # print('index:', index)
            for i in index:
                if (i >= 0) and (i <= FLAGS.sentence_len - 1):
                    k = x_title[i]
                    title = vocab.reverse(k)
                    # print(title + ' ')
                    f.write(title + ' ')
                elif (i >= FLAGS.sentence_len) and (i <= 2 * FLAGS.sentence_len - 2):
                    k1 = x_title[i - FLAGS.sentence_len]
                    k2 = x_title[i - FLAGS.sentence_len + 1]
                    title1 = vocab.reverse(k1)
                    title2 = vocab.reverse(k2)
                    # print(title1 + '-' + title2 + '  ')
                    f.write(title1 + '-' + title2 + '  ')
                else:
                    k1 = x_title[i - 2 * FLAGS.sentence_len + 1]
                    k2 = x_title[i - 2 * FLAGS.sentence_len + 2]
                    k3 = x_title[i - 2 * FLAGS.sentence_len + 3]
                    title1 = vocab.reverse(k1)
                    title2 = vocab.reverse(k2)
                    title3 = vocab.reverse(k3)
                    # print(title1 + '+' + title2 + '+' + title3 + '  ')
                    f.write(title1 + '+' + title2 + '+' + title3 + '  ')
            # f.write(str(label) + '\n')
            f.write('\n')


# 训练与测试
def train_test(isTrain=False, checkpoint_dir=None):
    # 1.建立textModel模型
    print('----------------------------------------------')
    print('Text model......')
    graph = tf.Graph()
    with graph.as_default():
        textModel = TextCNN(filter_sizes, FLAGS.num_filters, FLAGS.label_number, FLAGS.learning_rate,
                            FLAGS.batch_size,
                            FLAGS.decay_steps,
                            FLAGS.decay_rate, FLAGS.sentence_len, FLAGS.vocab_size, FLAGS.embed_size,
                            FLAGS.is_training,
                            multi_label_flag=FLAGS.multi_label_flag)

        # loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=textModel.logits, labels=textModel.input_y_multilabel))  # 原始loss
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=textModel.logits, labels=textModel.input_y_multilabel)
        # mask = tf.sequence_mask(FLAGS.sentence_len)  # 添加mask机制
        FLAGS.sentence_len = tf.placeholder(tf.int32, shape=[None])
        mask = tf.sequence_mask(FLAGS.sentence_len)  # 添加mask机制
        losses = tf.boolean_mask(loss, mask)
        loss = tf.reduce_mean(losses)

        optimizer = tf.train.AdamOptimizer(learning_rate=textModel.learning_rate).minimize(
            loss)  # trick: 先adam大学习率，之后sgd小学习率
        prediction = tf.argmax(tf.nn.softmax(textModel.logits), 1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(textModel.input_y_multilabel, 1)), tf.float32))
    # time.sleep(600)

    # 2.导入数据
    print('----------------------------------------------')
    print('load data......')
    # x_train, y_train, x_dev, y_dev, x_test, y_test, vocab = get_data()  # 从npy文件中导入数据（速度快）
    x_train, y_train, x_dev, y_dev, x_test, y_test, vocab = get_train_and_test_data()  # 从原始文件中导入数据（速度慢）

    # 3.开始训练/测试过程
    with tf.Session(graph=graph) as sess:
        saver = tf.train.Saver()
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        if isTrain:
            print('----------------------------------------------')
            print('start training......')
            if not os.listdir(checkpoint_dir):
                sess.run(tf.global_variables_initializer())
            else:
                print('load latest model......')
                ckpt = tf.train.latest_checkpoint(checkpoint_dir)
                print(ckpt)
                saver.restore(sess, ckpt)
            counter = 0
            f = open('../output/result/output.txt', 'a')
            for epoch in range(FLAGS.num_epochs):
                print("Epoch %d start !" % epoch)
                for x_batch, y_batch in fill_feed_dict(x_train, y_train, FLAGS.batch_size):
                    fd = {textModel.input_x: x_batch, textModel.input_y_multilabel: y_batch,
                          textModel.dropout_keep_prob: 0.5}
                    l, _, acc, alpha = sess.run([loss, optimizer, accuracy, textModel.alphas], feed_dict=fd)
                    # l, _, acc = sess.run([loss, optimizer, accuracy], feed_dict=fd)

                    counter += 1
                    if epoch == FLAGS.num_epochs - 4:
                        train_write(alpha, x_batch, y_batch, vocab, f)

                    if counter % 500 == 0:
                        print("Epoch %d\tcounter %d\tTrain Loss:%.3f\tacc:%.5f" % (epoch, counter, l, acc))

                    if counter % 20000 == 0:
                        for file in os.listdir(checkpoint_dir):
                            model_path = os.path.join(checkpoint_dir, file)
                            os.remove(model_path)
                        saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=counter)

                print("Validation accuracy and loss: ", sess.run([accuracy, loss], feed_dict={
                    textModel.input_x: x_dev,
                    textModel.input_y_multilabel: y_dev,
                    textModel.dropout_keep_prob: 1
                }))
            f.close()
            print("start predicting:  ")
            prediction, test_accuracy = sess.run([prediction, accuracy],
                                                 feed_dict={textModel.input_x: x_test,
                                                            textModel.input_y_multilabel: y_test,
                                                            textModel.dropout_keep_prob: 1})
            print('prediction:', prediction, len(prediction))
            print('test_acc:', test_accuracy)
            print("Test accuracy : %f %%" % (test_accuracy * 100))
        else:
            print('----------------------------------------------')
            print('start testing......')
            ckpt = tf.train.latest_checkpoint(checkpoint_dir)
            print('ckpt:', ckpt)
            saver.restore(sess, ckpt)
            if len(y_test) == 0:
                prediction = sess.run([textModel.predictions],
                                      feed_dict={textModel.input_x: x_test, textModel.dropout_keep_prob: 1})
                print(prediction[0], type(prediction[0]))
                return prediction[0]
            else:
                prediction, test_accuracy, alpha = sess.run([prediction, accuracy, textModel.alphas],
                                                            feed_dict={textModel.input_x: x_test,
                                                                       textModel.input_y_multilabel: y_test,
                                                                       textModel.dropout_keep_prob: 1})
                print("Test accuracy : %f %%" % (test_accuracy * 100))
                # test_write(alpha, x_test, y_test, vocab)
                print(x_test[0], prediction[0])
                return prediction


if __name__ == '__main__':
    train_test(isTrain=True, checkpoint_dir='../output/model/checkpoints_avgpool/')
    # train_test(isTrain=False, checkpoint_dir='../output/model/checkpoints_avgpool/')
