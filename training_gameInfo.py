import cv2
import tensorflow as tf
import numpy as np
import os
import sys
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    """
    This is only a function to indicate progress and does nothing else
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '|' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def get_batch(allData1, allData2, targetLen):
    all_data_shape = allData1.shape
    random_index = np.random.choice( all_data_shape[0], targetLen )
    res1 = np.zeros_like( allData1 )
    res2 = np.zeros_like( allData2 )
    for i in range(0, targetLen):
        res1[i] = allData1[random_index[i]]
        res2[i] = allData2[random_index[i]]
    return [res1[0:targetLen], res2[0:targetLen]]

def load_training_data():
    data_size = 100 # 16 48 68
    data = np.zeros( (data_size, 128, 128), dtype=np.float32 )
    label = np.zeros( (data_size, 8), dtype=np.float32 )
    f = open('training_data/label.txt', 'r')
    flines = f.readlines()
    for i in range(0, 100):
        imgPath = 'training_data/' + str(i) + '.png'
        tmpImg = cv2.imread(imgPath)
        tmpImg = cv2.cvtColor(tmpImg, cv2.COLOR_BGR2GRAY)
        tmpImg = cv2.resize(tmpImg, (128,128))
        data[i, :, :] = tmpImg[:, :]/255.0
        farray = flines[i].split(':')
        darray = farray[1].split(' ')
        for j in range(0,8):
            if j == 0 or j >= 2:
                label[i, j] = int(darray[j]) / 400.0
            else:
                label[i, j] = int(darray[j]) / 600.0
        printProgress(i+1, 100)
    return [data, label]

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def main():
    [d, l] = load_training_data()
    x = tf.placeholder(tf.float32, shape=[None, 128, 128])
    y_ = tf.placeholder(tf.float32, shape=[None, 8])
    x1 = tf.reshape(x, shape=[-1, 128, 128, 1])
    conv1 = tf.layers.conv2d(x1, 32, 5, activation=tf.nn.relu, padding='same')
    pool1 = tf.layers.max_pooling2d(conv1, strides=(2,2), pool_size=(2,2), padding='same')
    conv2 = tf.layers.conv2d(pool1, 64, 3, activation=tf.nn.relu, padding='same')
    pool2 = tf.layers.max_pooling2d(conv2, strides=(2,2), pool_size=(2,2), padding='same')
    conv3 = tf.layers.conv2d(pool2, 32, 3, activation=tf.nn.relu, padding='same')
    pool3 = tf.layers.max_pooling2d(conv3, strides=(2,2), pool_size=(2,2), padding='same')
    W_fc1 = weight_variable([16*16*32, 512])
    b_fc1 = bias_variable([512])
    h_pool2_flat = tf.reshape(pool3, [-1, 16*16*32])
    h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    W_fc2 = weight_variable([512, 8])
    b_fc2 = bias_variable([8])
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    loss = tf.reduce_mean(tf.square(y_conv - y_)) * 1000
    train_step =  tf.train.AdamOptimizer(1e-4).minimize(loss)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('training ')
        for i in range(20000):
            [x_batch, y_batch] = get_batch(d, l, 20)
            train_step.run(feed_dict={x: x_batch, y_: y_batch})
            printProgress(i+1, 20000)
            if i % 50 == 0:
                print('loss = ' + str( loss.eval(feed_dict={x: x_batch, y_: y_batch}) ))
        saver.save(sess, 'saved/model.ckpt')
        print('trained model saved. ')

if __name__ == '__main__':
    main()
