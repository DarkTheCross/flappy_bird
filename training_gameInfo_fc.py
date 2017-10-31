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
    data_size = 300 # 16 48 68
    data = np.zeros( (data_size, 128, 128), dtype=np.float32)
    label = np.zeros( (data_size, 8), dtype=np.float32)
    f = open('training_data/label.txt', 'r')
    flines = f.readlines()
    for i in range(0, 300):
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
        printProgress(i+1, 300)
    return [data, label]

def main():
    [d, l] = load_training_data()
    x = tf.placeholder(tf.float32, shape=[None, 128, 128], name="X" )
    y_ = tf.placeholder(tf.float32, shape=[None, 8], name='Y_' )
    x1 = tf.reshape(x, shape=[-1, 128, 128, 1])
    conv1 = tf.layers.conv2d(x1, 32, 5, activation=tf.nn.relu, padding='same')
    pool1 = tf.layers.max_pooling2d(conv1, strides=(2,2), pool_size=(2,2), padding='same')
    conv2 = tf.layers.conv2d(pool1, 64, 3, activation=tf.nn.relu, padding='same')
    pool2 = tf.layers.max_pooling2d(conv2, strides=(2,2), pool_size=(2,2), padding='same')
    conv3 = tf.layers.conv2d(pool2, 32, 3, activation=tf.nn.relu, padding='same')
    pool3 = tf.layers.max_pooling2d(conv3, strides=(2,2), pool_size=(2,2), padding='same')
    pool3_flatten = tf.contrib.layers.flatten(pool3)
    fc1 = tf.contrib.layers.fully_connected(pool3_flatten, 512, activation_fn=None)
    y_conv = tf.contrib.layers.fully_connected(fc1, 8, activation_fn=None)
    loss = tf.reduce_mean(tf.square(y_conv - y_)) * 1000
    train_step =  tf.train.AdamOptimizer(1e-4).minimize(loss)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('training ')
        for i in range(10000):
            [x_batch, y_batch] = get_batch(d, l, 20)
            train_step.run(feed_dict={x: x_batch, y_: y_batch})
            printProgress(i+1, 10000)
            if i % 50 == 0:
                print('loss = ' + str( loss.eval(feed_dict={x: x_batch, y_: y_batch}) ))
        saver.save(sess, 'saved/model.ckpt')
        print('trained model saved. ')

if __name__ == '__main__':
    main()
