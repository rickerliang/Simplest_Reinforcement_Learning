from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam
from keras.layers.advanced_activations import LeakyReLU

def build_model():
    print("modeling")
    model = Sequential()
    model.add(Dense(128, init='lecun_uniform', input_shape=(64,)))
    model.add(LeakyReLU(alpha=0.01))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5)) #I'm not using dropout, but maybe you wanna give it a try?

    model.add(Dense(128, init='lecun_uniform'))
    model.add(LeakyReLU(alpha=0.01))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(64, init='lecun_uniform'))
    model.add(LeakyReLU(alpha=0.01))

    model.add(Dense(4, init='lecun_uniform'))
    model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

    #rms = RMSprop()
    adam = Adam()
    model.compile(loss='mse', optimizer=adam)
    return model
