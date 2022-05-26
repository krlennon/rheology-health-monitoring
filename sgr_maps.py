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

def sgr_maps_ann(X_train_1, X_train_3, Y_train, X_test_1, X_test_3):
    # Define the two input layers
    inputg1 = Input(shape=(X_train_1.shape[1],))
    inputg3 = Input(shape=(X_train_3.shape[1],))

    # Create a network for the nonlinear response + LR
    xg3 = concatenate([inputg1, inputg3])
    xg3 = Dense(8, activation='relu', kernel_regularizer='l1')(xg3)
    z = Dense(5, activation='linear')(xg3)

    # Define the model
    model = Model(inputs=[inputg1, inputg3], outputs=z)

    # Train
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit([X_train_1, X_train_3], Y_train, epochs=6, steps_per_epoch=30000)
    model.save("model_maps")

    # Apply to data
    Y_pred = model.predict([X_test_1, X_test_3])
    Y_pred[:,2] = Y_pred[:,2]/10

    # Plot the parity for each predicted output
    for n in range(0,5):
        fig, ax = plt.subplots(1,1)
        t = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        ax.plot(t, Y_pred[:,n], ls='', marker='o', mfc='none', mec='b')

    plt.show()

# Import training data
all_data = sio.loadmat("data/synthetic/SGR_data_tensorial_569.mat")
stress_data = all_data["data_stress"]
J1 = stress_data[:,4:7]
J3_r = stress_data[:,7:26]

# For the tensorial SGR
J3_tss = stress_data[:,26:]
Cv = 10*np.random.rand(10000,1)
Dv = -1*np.random.rand(10000,1)
J3 = J3_r * Cv + J3_tss * Dv
Y_train = np.concatenate((np.log(stress_data[:,0:2]), 10*np.transpose(np.array([stress_data[:,2]])),
    np.array(Cv), np.array(Dv)), axis=1)

# Filter only the imaginary part (real part is very noisy)
J1_train = np.imag(J1)
J3_train = np.imag(J3)

# Add some error to the data
eps_1 = 0.1 # relative error
eps_3 = 1 # absolute error
J1_train += eps_1*np.random.randn(10000,J1_train.shape[1])*J1_train
J3_train += eps_3*np.random.randn(10000,J3_train.shape[1])

# Take the absolute value
X_train_1 = np.log(np.abs(J1_train))
X_train_3 = np.log(np.abs(J3_train))

# Get the test data
X_test = np.load("maps_data.npy")
J1_test = np.imag(X_test[:,:3])
J3_test = np.imag(X_test[:,3:])
X_test_1 = np.log(np.abs(J1_test))
X_test_3 = np.log(np.abs(J3_test))

# Select task
sgr_maps_ann(matlib.repmat(X_train_1,10,1),matlib.repmat(X_train_3,10,1),matlib.repmat(Y_train,10,1),X_test_1,X_test_3)

