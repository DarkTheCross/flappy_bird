import tensorflow as tf
import os
import numpy as np
import cv2
from FBGame import FBGame
import random
from conv2d import conv2d
import skimage.measure

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class GameInfoExtractor:
    def __init__(self):
        sess = tf.Session()
        export_dir = 'saveModelDir/'
        tf.saved_model.loader.load(sess, ['gameInfo'], export_dir)
        self.c0k = sess.run('conv2d/kernel:0')
        self.c0b = sess.run('conv2d/bias:0')
        self.c1k = sess.run('conv2d_1/kernel:0')
        self.c1b = sess.run('conv2d_1/bias:0')
        self.c2k = sess.run('conv2d_2/kernel:0')
        self.c2b = sess.run('conv2d_2/bias:0')
        self.fc0w = sess.run('fully_connected/weights:0')
        self.fc0b = sess.run('fully_connected/biases:0')
        self.fc1w = sess.run('fully_connected_1/weights:0')
        self.fc1b = sess.run('fully_connected_1/biases:0')

    def preprocess_game_scene(self, gameScene):
        data = np.zeros( (128, 128), dtype=np.float32 )
        tmpImg = cv2.cvtColor(gameScene, cv2.COLOR_BGR2GRAY)
        tmpImg = cv2.resize(tmpImg, (128,128))
        data = tmpImg/255.0
        return data

    def conv2dWithRelu(self, img, k, b):
        r1 = conv2d(img, k)
        for i in range(0, r1.shape[3]):
            r1[:, :, :, i] += b[i]
        r1 = np.maximum(r1, 0, r1)
        return r1

    def fc(self, x, w, b):
        return np.add(np.matmul(x, w), b)

    def evaluateGameScene(self, gameScene):
        s = np.array([self.preprocess_game_scene(gameScene)])
        rs = np.resize(s, [s.shape[0], s.shape[1], s.shape[2], 1])
        conv1 = self.conv2dWithRelu(rs, self.c0k, self.c0b)
        pool1 = skimage.measure.block_reduce(conv1, (1, 2, 2, 1), np.max)
        conv2 = self.conv2dWithRelu(pool1, self.c1k, self.c1b)
        pool2 = skimage.measure.block_reduce(conv2, (1, 2, 2, 1), np.max)
        conv3 = self.conv2dWithRelu(pool2, self.c2k, self.c2b)
        pool3 = pool2 = skimage.measure.block_reduce(conv3, (1, 2, 2, 1), np.max)
        pool3_flatten = pool3.flatten()
        fc1 = self.fc(pool3_flatten, self.fc0w, self.fc0b)
        fc2 = self.fc(fc1, self.fc1w, self.fc1b)
        return fc2
