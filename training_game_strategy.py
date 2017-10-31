import tensorflow as tf
import os
import numpy as np
import cv2
from FBGame import FBGame
import random
import sys
#from GameInfoExtractor import GameInfoExtractor

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
    #ext = GameInfoExtractor()
    with tf.Session() as sess:
        export_dir = 'saveModelDir/'
        tf.saved_model.loader.load(sess, ['gameInfo'], export_dir)
        x = tf.placeholder(tf.float32, shape=[None, 20])
        y_ = tf.placeholder(tf.float32, shape=[None, 2])
        fc1 = tf.contrib.layers.fully_connected(x, 512, activation_fn=tf.nn.relu)
        fc2 = tf.contrib.layers.fully_connected(fc1, 512, activation_fn=tf.nn.relu)
        y_conv = tf.contrib.layers.fully_connected(fc2, 2, activation_fn=None)
        loss = tf.reduce_sum(tf.square(y_conv - y_)) / 10
        trainer = tf.train.GradientDescentOptimizer(0.1)
        updateModel = trainer.minimize(loss)
        saver = tf.train.Saver()
        lamb = 0.95
        expl = 1
        sess.run(tf.global_variables_initializer())
        g = FBGame()
        g.load_resources()
        s = g.render_scene()
        game_state_rec = np.zeros((20,), dtype=np.float32)
        game_next_rec = np.zeros((20,), dtype=np.float32)
        rep_mem = np.zeros((40, 42), dtype=np.float32)
        for i in range(1, 10000):
            act = int(random.random()>0.9  )
            r, t, s = g.update_scene(act)
            expl = max((1-float(i)/2000), 0.1)
            while r:
                game_state_rec[:] = game_next_rec[:]
                Q_tmp = y_conv.eval(feed_dict={x: [game_state_rec]})
                p = bool(random.random()<expl)
                if p:
                    act = int(random.random()>0.9  )
                else:
                    act = np.argmax(Q_tmp)
                r, t, s = g.update_scene(act)
                tmpx_next = sess.run('fully_connected_1/BiasAdd:0', feed_dict = { 'X:0': [preprocess_game_scene(s)] })
                game_next_rec[:16] = game_state_rec[4:]
                game_next_rec[16:] = tmpx_next[0, :4]

                rep_mem[:39, :] = rep_mem[1:, :]
                rep_mem[39, :20] = game_state_rec[:]
                rep_mem[39, 20] = act
                rep_mem[39, 21] = t
                rep_mem[39, 22:] = game_next_rec[:]

            '''
            for j in range(0, 39):
                if not rep_mem[38-j, 21] == 1:
                    rep_mem[38-j, 21] = rep_mem[39-j, 21]
            '''
            #print(rep_mem[:, 21])

            rc = np.random.choice(40, 15)
            batch_s = np.zeros((15, 20), dtype=np.float32)
            batch_y = np.zeros((15, 2), dtype=np.float32)

            for j in range(0, 15):
                batch_s[j, :] = rep_mem[rc[j], :20]
                tmpRew = y_conv.eval(feed_dict={x: [rep_mem[rc[j], 22:]]})
                am = np.max(tmpRew)
                batch_y[j, int(rep_mem[rc[j], 20])] = rep_mem[rc[j], 21] + lamb * am

            sess.run(updateModel, feed_dict = {x: batch_s, y_: batch_y})

            l = sess.run(loss, feed_dict = {x: batch_s, y_: batch_y})
            print(i, l)
            game_state_rec = np.zeros((20,), dtype=np.float32)
            game_next_rec = np.zeros((20,), dtype=np.float32)
            rep_mem = np.zeros((40, 42), dtype=np.float32)
            g.__init__()
            printProgress(i, 10000)
        saver.save(sess, 'str_saved/model.ckpt')

if __name__ == '__main__':
    main()
