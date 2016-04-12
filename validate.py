from gridworld import initGrid, makeMove, getReward, dispGrid, initGridRand, initGridPlayer
from networkmodel import build_model
import drawgridworld
from keras.models import Sequential
import numpy as np
import random
import time

def testAlgo(init=0):
    arrow = ["^", "v", "<", ">"]
    i = 0
    if init==0:
        state = initGrid()
    elif init==1:
        state = initGridPlayer()
    else:
        state = initGridRand()

    drawgridworld.draw_state(state, 0)
    time.sleep(0.5)
    #print("Initial State:")
    #print(dispGrid(state))
    status = 1
    #while game still in progress
    while(status == 1):
        qval = model.predict(state.reshape(1,64), batch_size=1)
        action = (np.argmax(qval)) #take action with highest Q-value
        print('Move #: %s; Taking action: %s' % (i, arrow[action]))
        state = makeMove(state, action)[0]
        #print(dispGrid(state))
        reward = getReward(state)
        if reward != -1:
            status = 0
            print("Reward: %s" % (reward,))
        i += 1 #If we're taking more than 10 actions, just stop, we probably can't win this game
        drawgridworld.draw_state(state, i - 10)
        time.sleep(0.1)
        if (i > 10):
            print("Game lost; too many moves.")
            reward = -10
            break
    time.sleep(0.5)
    return reward

def validate():
    valid = 0;
    for j in range(100):
        reward = testAlgo(init=2) # init => game mode
        print(reward)
        if (reward > 0):
            valid += reward
    print(valid)

model = build_model()
model.load_weights("./model_weight.complete")
validate()
