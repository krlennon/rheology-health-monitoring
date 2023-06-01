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
import keras.models

def nll_gumbel(y_true, y_pred):
    # Negative log-likelihood for a Gumbel distribution
    z = tf.divide(y_true, tf.square(y_pred))
    nll = z + tf.math.exp(tf.math.scalar_mul(-1, z)) + tf.math.log(tf.square(y_pred))
    return tf.reduce_mean(nll, axis=-1)

def sgr_lr_unc(X_train,Y_train,X_test,Y_test):
    # Construct a fully-connected ANN 
    model = Sequential()

    # Input layer
    model.add(Dense(12, activation='relu', input_dim=X_train.shape[1]))

    # Hidden layers
    layers = 1
    for n in range(0,layers):
        model.add(Dense(9, activation='relu'))

    # Output layer
    model.add(Dense(3, activation='linear'))

    # Compile, train, predict
    model.compile(optimizer='rmsprop', loss=nll_gumbel)
    model.fit(X_train, Y_train, epochs=5, steps_per_epoch=10000)
    model.save("model_synthetic_beta_lr")
    Y_pred = model.predict(X_test)

    for n in range(0,3):
        fig, ax = plt.subplots(1,1)
        ax.errorbar(Y_test[:,n] + X_test[:,n], X_test[:,n], yerr=(Y_pred[:,n])**2, ls='', marker='o', mfc='none', mec='b')

    plt.show()

# Import training data
alldata = sio.loadmat("data/synthetic/SGR_data_tensorial_569.mat")
stress_data = alldata["data_stress"]
J1 = np.imag(stress_data[:,4:7])
Y = np.real(np.concatenate((np.log(stress_data[:,0:2]), 10*np.transpose(np.array([stress_data[:,2]]))),axis=1))

# Add some error to the data
J1 = np.log(np.abs(J1))

# Load trained model and predict
model_lr = keras.models.load_model("model_synthetic_mean_lr")
Y_pred = model_lr.predict(J1)
Y_diff = Y - Y_pred

# Split data to testing and training sets
X_train = Y_pred[:9000,:]
X_test = Y_pred[9000:,:]
Y_train = Y_diff[:9000,:]
Y_test = Y_diff[9000:,:]

# Fit to the ANN
sgr_lr_unc(matlib.repmat(X_train,10,1),matlib.repmat(Y_train,10,1),X_test,Y_test)
