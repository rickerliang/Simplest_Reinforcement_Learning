from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam
from keras.layers.advanced_activations import LeakyReLU
from theano import tensor as T

huber_delta = 1.0

def huber_loss(y_true, y_pred):
    diff = (y_true - y_pred)
    diff_abs = T.abs_(diff)
    square_targets = diff[T.lt(diff_abs, huber_delta)]
    linear_targets = diff[T.ge(diff_abs, huber_delta)]
    square_loss = ((square_targets ** 2.0) * 0.5).sum()
    linear_loss = ((T.abs_(linear_targets) + (-0.5 * huber_delta)) * huber_delta).sum()
    return T.mean((square_loss + linear_targets), axis = -1)

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

    #rms = RMSprop()
    adam = Adam()
    model.compile(loss=huber_loss, optimizer=adam)
    return model
