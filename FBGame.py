import cv2
import numpy as np
import random

import sys

import matplotlib.pyplot as plt


def paste(scene_img, object_img, left_top_point):
    # calculate valid region
    so = object_img.shape
    ss = scene_img.shape
    left_top_point = (int(left_top_point[1]), int(left_top_point[0]))
    xi = max(0, left_top_point[0])
    xa = min(ss[0], left_top_point[0] + so[0])
    yi = max(0, left_top_point[1])
    ya = min(ss[1], left_top_point[1] + so[1])
    if yi >= ya or xi > xa:
        return scene_img
    alpha = object_img[:, :, 3]/255.0
    s_alpha = 1.0-alpha
    #print(xi, xa, yi, ya)
    for c in range(0,3):
        scene_img[xi:xa, yi:ya, c] = object_img[ xi-left_top_point[0] : xa - left_top_point[0], yi - left_top_point[1] : ya - left_top_point[1], c] * alpha[ xi-left_top_point[0] : xa - left_top_point[0], yi - left_top_point[1] : ya - left_top_point[1]] + scene_img[xi:xa, yi:ya, c] * s_alpha[ xi-left_top_point[0] : xa - left_top_point[0], yi - left_top_point[1] : ya - left_top_point[1]]
    return scene_img

class FBGame:
    def __init__(self):
        self.parameters = { 'window_width': 600, 'window_height': 400, 'g': 800, 'a_y': -200, 'v': 200, 'bird_x': 50,'bird_width': 25, 'bird_height': 25,
         'block_width': 40, 'block_gap': 200, 'block_top_thresh': 100, 'block_min_size': 70, 'block_bot_thresh': 300, 'block_image_height': 400, 'fps': 30.0 }
        self.gameState = { 'bird_y': 200.0, 'v_y': 0.0, 'dist_to_next_block': 300.0}
        self.gameState['blocks'] = []
        for i in range(0,4):
            tmpTop = random.randint(self.parameters['block_top_thresh'], self.parameters['block_bot_thresh'] - self.parameters['block_min_size'])
            tmpBot = random.randint(tmpTop + self.parameters['block_min_size'], self.parameters['block_bot_thresh'])
            self.gameState['blocks'].append([tmpTop, tmpBot])
        self.running = True
        self.score = 0

    def load_resources(self):
        self.images = {}
        self.images['birdImg'] = cv2.imread('imgs/flappy-base.png', cv2.IMREAD_UNCHANGED)
        self.images['bgImg'] = cv2.imread('imgs/background.png', cv2.IMREAD_UNCHANGED)
        self.images['upperBlockImg'] = cv2.imread('imgs/upper-pillar.png', cv2.IMREAD_UNCHANGED)
        self.images['lowerBlockImg'] = cv2.imread('imgs/lower-pillar.png', cv2.IMREAD_UNCHANGED)
        self.images['birdImg'] = cv2.resize(self.images['birdImg'], (self.parameters['bird_width'], self.parameters['bird_height']))
        self.images['bgImg'] = cv2.resize(self.images['bgImg'], (self.parameters['window_width'], self.parameters['window_height']))
        [h1, w1] = self.images['upperBlockImg'].shape[:2]
        self.images['upperBlockImg'] = cv2.resize(self.images['upperBlockImg'], (self.parameters['block_width'], h1))
        [h2, w2] = self.images['lowerBlockImg'].shape[:2]
        self.images['lowerBlockImg'] = cv2.resize(self.images['lowerBlockImg'], (self.parameters['block_width'], h2))

    def render_scene(self):
        self.scene = self.images['bgImg'].copy()
        paste( self.scene, self.images['birdImg'], ( self.parameters['bird_x'] - self.parameters['bird_width']/2, self.gameState['bird_y'] - self.parameters['bird_height']/2 ))
        for idx, b in enumerate(self.gameState['blocks']):
            paste( self.scene, self.images['upperBlockImg'], ( self.parameters['bird_x'] + self.gameState['dist_to_next_block'] + idx * self.parameters['block_gap'], b[0] - self.parameters['block_image_height'] ))
            paste( self.scene, self.images['lowerBlockImg'], ( self.parameters['bird_x'] + self.gameState['dist_to_next_block'] + idx * self.parameters['block_gap'], b[1] ))

    def update_scene(self, up):
        if self.running == False:
            return
        time_interval = 1/self.parameters['fps']
        self.gameState['v_y'] += self.parameters['g'] * time_interval
        self.gameState['bird_y'] += self.gameState['v_y'] * time_interval
        self.gameState['dist_to_next_block'] -= self.parameters['v'] * time_interval
        if up:
            self.gameState['v_y'] = self.parameters['a_y']
        if self.gameState['dist_to_next_block'] + self.parameters['block_width'] < 0:
            self.gameState['dist_to_next_block'] += self.parameters['block_gap']
            self.gameState['blocks'].pop(0)
            tmpTop = random.randint(self.parameters['block_top_thresh'], self.parameters['block_bot_thresh'] - self.parameters['block_min_size'])
            tmpBot = random.randint(tmpTop + self.parameters['block_min_size'], self.parameters['block_bot_thresh'])
            self.gameState['blocks'].append([tmpTop, tmpBot])
        if self.gameState['dist_to_next_block'] < self.parameters['bird_width']/2 and self.gameState['dist_to_next_block'] + self.parameters['block_width'] > - self.parameters['bird_width']/2 :
            if self.gameState['bird_y'] + self.parameters['bird_height'] > self.gameState['blocks'][0][1] or self.gameState['bird_y'] - self.parameters['bird_height'] < self.gameState['blocks'][0][0]:
                self.running = False # game over
                print('Game Over')
                self.score = -2
        self.score += 1
        self.render_scene()
        return self.score, self.scene

    def generate_random_game_scene(self, file_name):
        self.gameState['dist_to_next_block'] = random.randint(-self.parameters['block_width'], self.parameters['block_gap'])
        self.gameState['blocks'] = []
        for i in range(0,4):
            tmpTop = random.randint(self.parameters['block_top_thresh'], self.parameters['block_bot_thresh'] - self.parameters['block_min_size'])
            tmpBot = random.randint(tmpTop + self.parameters['block_min_size'], self.parameters['block_bot_thresh'])
            self.gameState['blocks'].append([tmpTop, tmpBot])
        if self.gameState['dist_to_next_block'] < self.parameters['bird_width']/2 and self.gameState['dist_to_next_block'] + self.parameters['block_width'] > - self.parameters['bird_width']/2 :
            self.gameState['bird_y'] = random.randint(int(self.gameState['blocks'][0][0] + self.parameters['bird_height']/2), int(self.gameState['blocks'][0][1] - self.parameters['bird_height']/2))
        else:
            self.gameState['bird_y'] = random.randint(20, self.parameters['window_height'] - 20)
        self.render_scene()
        imgStr = 'training_data/' + file_name + '.png'
        print(imgStr)
        cv2.imwrite( imgStr, self.scene )
        return self.gameState

    def press(self, event):
        #print('press', event.key)
        sys.stdout.flush()
        if event.key == 'escape':
            self.running = False
        elif event.key == ' ':
            self.keyDown = True
        elif event.key == 'enter':
            self.__init__()
            self.running = True

    def run(self):
        plt.ion()
        self.running = True
        self.keyDown = False
        fig, ax = plt.subplots()
        fig.canvas.mpl_connect('key_press_event', self.press)
        ax.imshow(cv2.cvtColor(self.scene, cv2.COLOR_BGR2RGB))
        plt.show()
        while(1):
            ax.clear()
            ax.imshow(cv2.cvtColor(self.scene, cv2.COLOR_BGR2RGB))
            fig.canvas.draw()
            plt.pause(0.01)
            if self.running:
                self.update_scene(self.keyDown)
                self.keyDown = False
