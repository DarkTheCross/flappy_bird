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

def main():
    x = tf.placeholder(tf.float32, shape=[None, 128, 128])
    y_ = tf.placeholder(tf.float32, shape=[None, 8])
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
        saver.restore(sess, 'saved/model.ckpt')
        p = tf.all_variables()
        print(p)
        g = FBGame()
        g.load_resources()
        g.render_scene()
        for i in range(1, 500):
            p = bool(random.random()>0.9)
            r, t, s = g.update_scene(p)
            tmpx = preprocess_game_scene(s)
            tmpres = y_conv.eval(feed_dict={x: [tmpx]})
            print(tmpres)
            print(t)
            if not r:
                break

if __name__ == '__main__':
    main()
