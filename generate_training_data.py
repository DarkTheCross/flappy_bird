from FBGame import FBGame

g = FBGame()
g.load_resources()

f = open('training_data/label.txt', 'w')

for i in range(0, 100):
    stateRec = g.generate_random_game_scene( str(i) )
    line = str(i) + ':'
    line += str(stateRec['bird_y'])
    line += ' '
    line += str(stateRec['dist_to_next_block'])
    line += ' '
    for c in range(0,3):
        line += str(stateRec['blocks'][c][0])
        line += ' '
        line += str(stateRec['blocks'][c][1])
        line += ' '
    line += '\n'
    f.write(line)

f.close()
