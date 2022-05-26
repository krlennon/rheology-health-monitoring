import numpy as np
import numpy.matlib as matlib
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.markers
from keras.models import Model, Sequential
from keras.layers import Input, Dense, add, Lambda, concatenate
import keras.backend as K
import tensorflow as tf
from keras import regularizers
import pandas as pd

def sgr_lr_ann(X_train,Y_train,X_test,Y_test=None):
    # Construct a fully-connected ANN 
    model = Sequential()

    # Input layer
    model.add(Dense(16, activation='relu', input_dim=8))

    # Hidden layers
    layers = 2
    for n in range(0,layers):
        model.add(Dense(16, activation='relu'))

    # Output layer
    model.add(Dense(3, activation='linear'))

    # Compile, train, predict
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(X_train, Y_train, epochs=4, steps_per_epoch=30000)
    model.save("model_lr")
    Y_pred = model.predict(X_test)

    fig, ax = plt.subplots(2,2)
    for n in range(0,3):
        i = np.floor_divide(n,2)
        j = n - 2*i
        ax[i,j].plot(Y_pred[:,n], 'o')

    plt.show()

# Import training data
alldata = sio.loadmat("data/synthetic/SGR_LR.mat")
data = alldata["data_strain"]
G1 = data[:,3:]
Y = np.concatenate((np.log(data[:,0:2]), 10*np.transpose(np.array([data[:,2]]))), axis=1)
X = np.imag(G1)

# Add some error to the data
eps = 0.001 # relative error
X = X + eps*np.random.randn(10000,X.shape[1])
X = np.log(np.abs(X))

# Reformat data to testing and training sets
X_train = X
Y_train = Y

# Import testing data
mins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
X_test = np.zeros((len(mins),X.shape[1]))
for i in range(len(mins)):
    df = pd.read_csv(f"data/saos/saos_{mins[i]}min.csv", header=None)
    data = df.to_numpy()
    Gp = np.flip(data[:,0])
    Gpp = np.flip(data[:,1])
    X_test[i,:] = Gpp
X_test = np.log(np.abs(X_test))

# Fit to the ANN
sgr_lr_ann(matlib.repmat(X_train,10,1),matlib.repmat(Y_train,10,1),X_test)
