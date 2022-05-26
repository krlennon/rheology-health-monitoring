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

def split_ANN(X_train_1, X_train_3, Y_train, X_test_1, X_test_3, Y_test):
    # Define the two input layers
    inputg1 = Input(shape=(12,))
    inputg3 = Input(shape=(76,))

    # Create a network for the linear response
    layers1 = 2 # hidden layers
    xg1 = Dense(12, activation='relu')(inputg1)
    for n in range(1,layers1):
        xg1 = Dense(9, activation='relu')(xg1)
    xg1 = Dense(5, activation='linear')(xg1)

    # Create a network for the nonlinear response + LR
    layers3 = 2 # hidden layers
    xg3 = concatenate([inputg1, inputg3])
    xg3 = Dense(88, activation='relu')(xg3)
    for n in range(1,layers3):
        xg3 = Dense(46, activation='relu')(xg3)
    z = Dense(5, activation='linear')(xg3)
    z = concatenate([z, xg1])

    # Define the model
    model = Model(inputs=[inputg1, inputg3], outputs=z)

    # Train
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit([X_train_1, X_train_3], Y_train, epochs=5, steps_per_epoch=30000)

    # Test
    Y_pred = model.predict([X_test_1, X_test_3])
    Y_pred[:,2] = Y_pred[:,2]/10
    Y_pred[:,7] = Y_pred[:,7]/10
    Y_test[:,2] = Y_test[:,2]/10
    Y_test[:,7] = Y_test[:,7]/10

    # Plot the parity for each predicted output
    for n in range(0,5):
        markerG3 = matplotlib.markers.MarkerStyle(marker='o')
        markerG1 = matplotlib.markers.MarkerStyle(marker='^')
        fig, ax = plt.subplots(1,1)
        ax.scatter(Y_test[:,n+5], Y_pred[:,n+5], marker=markerG1, facecolors='none', edgecolors='r')
        ax.scatter(Y_test[:,n], Y_pred[:,n], marker=markerG3, facecolors='none', edgecolors='b')

        # Add parity line
        left, right = ax.get_xlim()
        bottom, top = ax.get_ylim()
        lower = max([left,bottom])
        upper = min([right,top])
        ax.set_xlim([lower,upper])
        ax.set_ylim([lower,upper])
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=15)
        plt.plot([lower,upper],[lower,upper],'k--')

    # Print the rmse for the 'x' predictions
    rmse = np.sqrt(np.average(np.real(Y_pred[:,2] - Y_test[:,2])**2))

    plt.show()

# Import testing and training data
alldata = sio.loadmat("data/synthetic/SGR_data_tensorial_1416.mat")
data = alldata["data_strain"]
H1 = data[:,4:7]
H3_r = data[:,7:26]

# For the tensorial SGR
H3_tss = data[:,26:]
Cv = 10*np.random.rand(10000,1)
Dv = -1*np.random.rand(10000,1)
H3 = H3_r * Cv + H3_tss * Dv
Y = np.concatenate( ( np.log( data[:,0:2] ), 10*np.transpose( np.array( [data[:,2]] ) ),
    np.array( Cv ), np.array( Dv ) ), axis=1 )
H1 = np.append(np.real(H1),np.imag(H1),axis=1)
H3 = np.append(np.real(H3),np.imag(H3),axis=1)

# Add some error to the data
eps = 0.1 # relative error
H1 += eps*np.random.randn(10000,6)*H1
H3 += eps*np.random.randn(10000,38)*H3

# For the split network
X_1 = np.append(np.log(np.abs(H1)),np.sign(H1),axis=1)
X_3 = np.append(np.log(np.abs(H3)),np.sign(H3),axis=1)
X_train_1 = X_1[:9000,:]
X_test_1 = X_1[9000:,:]
X_train_3 = X_3[:9000,:]
X_test_3 = X_3[9000:,:]
Y_train = Y[:9000]
Y_test = Y[9000:]

# Select task
Y_train = np.append(Y_train,Y_train,axis=1)
Y_test = np.append(Y_test,Y_test,axis=1)
split_ANN(matlib.repmat(X_train_1,5,1),matlib.repmat(X_train_3,5,1),matlib.repmat(Y_train,5,1),
        matlib.repmat(X_test_1,1,1),matlib.repmat(X_test_3,1,1),matlib.repmat(Y_test,1,1))
