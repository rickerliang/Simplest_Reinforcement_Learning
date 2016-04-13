from gridworld import initGrid, makeMove, getReward, dispGrid, initGridRand, initGridPlayer
from networkmodel import build_model
from keras.models import Sequential
import numpy as np
import random
import time

replay = [] #stores tuples of (S, A, R, S')
model_weight_path = "./model.weight"
batchSize = 20
buffer = 40
model_hat_update_freq = 1000 # step
model_hat = build_model()
model = build_model()
model_hat.save_weights(model_weight_path, True)
model.load_weights(model_weight_path)
mini_batch_step = 0
retrain_limit = 50

def mini_batch():
    global mini_batch_step
    mini_batch_step += 1
    gamma = 0.975
    #randomly sample our experience replay memory
    minibatch = random.sample(replay, batchSize)
    x_train = []
    y_train = []
    for memory in minibatch:
        #Get max_Q(S',a)
        old_state, action, reward, new_state = memory
        old_qval = model_hat.predict(old_state.reshape(1,64), batch_size=1)
        newQ = model_hat.predict(new_state.reshape(1,64), batch_size=1)
        maxQ = np.max(newQ)
        y = np.zeros((1,4))
        y[:] = old_qval[:]
        if reward == -1: #non-terminal state
            update = (reward + (gamma * maxQ))
        else: #terminal state
            update = reward
        y[0][action] = update
        x_train.append(old_state.reshape(64,))
        y_train.append(y.reshape(4,))
    if ((mini_batch_step % 1000) > 0):
        model.fit(np.array(x_train), np.array(y_train), batch_size=batchSize, nb_epoch=1, verbose=0)
    else:
        # verbose
        model.fit(np.array(x_train), np.array(y_train), batch_size=batchSize, nb_epoch=1, verbose=1)

def train():
    print("training")

    epsilon_decay = 1.0/20
    epsilon = 1.0
    history_indexer = 0
    before_perf = 0
    i = 0
    retrain_count = 0

    while(True):
        i += 1
        if (i % model_hat_update_freq == 0):
            print("validate model performance")
            current_model_perf = validate()
            print("before %d - current %d" %(before_perf, current_model_perf))
            if (current_model_perf > before_perf):
                retrain_count = 0
                before_perf = current_model_perf
                print("update model_hat")
                print("epsilon %f" % epsilon)
                model.save_weights(model_weight_path, True)
                model_hat.load_weights(model_weight_path)
            else:
                retrain_count += 1
                if (retrain_count > retrain_limit):
                    print("retrain limit reached")
                    print(before_perf)
                    return
                print("retrain")
                if (epsilon > 0.2):
                    epsilon -= epsilon_decay
                elif (epsilon > 0.02):
                    epsilon -= epsilon_decay / 10.0
                print("epsilon %f" % epsilon)
                model.load_weights(model_weight_path)
        state = initGridRand() #using the harder state initialization function
        status = 1
        #while game still in progress
        while(status == 1):
            #We are in state S
            #Let's run our Q function on S to get Q values for all possible actions
            qval = model.predict(state.reshape(1,64), batch_size=1)
            if (random.random() < epsilon): #choose random action
                action = np.random.randint(0,4)
            else: #choose best action from Q(s,a) values
                action = (np.argmax(qval))
            #Take action, observe new state S'
            #print("action")
            #print(action)
            #print(dispGrid(state))
            new_state, hit_wall = makeMove(state, action)
            #Observe reward
            reward_current = getReward(new_state)
            if hit_wall:
                reward_current = -2
            #print("reward_a")
            #print(reward_current)
            #Experience replay storage
            if (len(replay) < buffer): #if buffer not filled, add to it
                replay.append((state, action, reward_current, new_state))
            else: #if buffer full, overwrite old values
                if (history_indexer < (buffer-1)):
                    history_indexer += 1
                else:
                    history_indexer = 0
                replay[history_indexer] = (state, action, reward_current, new_state)
                #print("Game #: %s" % (i,))
                mini_batch()                
            state = new_state
            if reward_current == -10 or reward_current == 10: #if reached terminal state, update game status
                status = 0
        #if epsilon > 0.1: #decrement epsilon over time
        #    epsilon -= epsilon_decay

def testAlgo(init=0):
    arrow = ["^", "v", "<", ">"]
    i = 0
    if init==0:
        state = initGrid()
    elif init==1:
        state = initGridPlayer()
    else:
        state = initGridRand()

    #print("Initial State:")
    #print(dispGrid(state))
    status = 1
    #while game still in progress
    while(status == 1):
        qval = model.predict(state.reshape(1,64), batch_size=1)
        action = (np.argmax(qval)) #take action with highest Q-value
        #print('Move #: %s; Taking action: %s' % (i, arrow[action]))
        state = makeMove(state, action)[0]
        #print(dispGrid(state))
        #time.sleep(0.1)
        reward = getReward(state)
        if reward != -1:
            status = 0
            #print("Reward: %s" % (reward,))
        i += 1 #If we're taking more than 10 actions, just stop, we probably can't win this game
        if (i > 10):
            #print("Game lost; too many moves.")
            reward = -10
            break
    return reward

def validate():
    valid = 0;
    for j in range(1000):
        reward = testAlgo(init=2)
        #print(reward)
        if (reward > 0):
            valid += reward
    print("validate reward %d" % valid)
    return valid

train()
validate()
model_hat.save_weights("./model_weight.complete", True)

