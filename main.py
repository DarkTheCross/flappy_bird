from FBGame import FBGame
import cv2
from GameInfoExtractor import GameInfoExtractor

g = GameInfoExtractor()
#g.saveModel()
#g.loadModel()

s = cv2.imread('training_data/2.png')
cv2.imshow('s', s)
cv2.waitKey(0)
r = g.evaluateGameScene(s)
print(r)
'''
g = FBGame()
g.load_resources()
g.render_scene()
g.run()
'''
