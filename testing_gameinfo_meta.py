import tensorflow as tf
import os
import numpy as np
import cv2
from FBGame import FBGame
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

def preprocess_game_scene(gameScene):
    data = np.zeros( (128, 128), dtype=np.float32 )
    tmpImg = cv2.cvtColor(gameScene, cv2.COLOR_BGR2GRAY)
    tmpImg = cv2.resize(tmpImg, (128,128))
    data = tmpImg/255.0
    return data

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def main():
    x = tf.placeholder(tf.float32, shape=[None, 128, 128])
    y_ = tf.placeholder(tf.float32, shape=[None, 8])
    x1 = tf.reshape(x, shape=[-1, 128, 128, 1])
    conv1 = tf.layers.conv2d(x1, 32, 5, activation=tf.nn.relu, padding='same', trainable = False)
    pool1 = tf.layers.max_pooling2d(conv1, strides=(2,2), pool_size=(2,2), padding='same')
    conv2 = tf.layers.conv2d(pool1, 64, 3, activation=tf.nn.relu, padding='same', trainable = False)
    pool2 = tf.layers.max_pooling2d(conv2, strides=(2,2), pool_size=(2,2), padding='same')
    conv3 = tf.layers.conv2d(pool2, 32, 3, activation=tf.nn.relu, padding='same', trainable = False)
    pool3 = tf.layers.max_pooling2d(conv3, strides=(2,2), pool_size=(2,2), padding='same')
    W_fc1 = weight_variable([16*16*32, 512])
    b_fc1 = bias_variable([512])
    h_pool2_flat = tf.reshape(pool3, [-1, 16*16*32])
    h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    W_fc2 = weight_variable([512, 8])
    b_fc2 = bias_variable([8])
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    loss = tf.reduce_mean(tf.square(y_conv - y_))
    train_step =  tf.train.AdamOptimizer(1e-4).minimize(loss)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'saved/model.ckpt')
        g = FBGame()
        g.load_resources()
        g.render_scene()
        for i in range(1, 50):
            p = bool(random.random()>0.9)
            s1, s2 = g.update_scene(p)
            print(s2)
            tmpx = preprocess_game_scene(s2)
            tmpres = y_conv.eval(feed_dict={x: [tmpx]})
            cv2.imshow('d', tmpx)
            print(tmpres)
            cv2.waitKey(0)

if __name__ == '__main__':
    main()
