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
import keras.models

def nll_gumbel(y_true, y_pred):
    # Negative log-likelihood for a Gumbel distribution
    z = tf.divide(y_true, tf.add(tf.square(y_pred), 1))
    nll = z + tf.math.exp(tf.math.scalar_mul(-1, z)) + tf.math.log(tf.add(tf.square(y_pred), 1))
    return tf.reduce_mean(nll, axis=-1)

def sgr_maps_unc(X_train, Y_train, X_test, Y_test):
    # Define the two input layers
    inputg1 = Input(shape=(X_train.shape[1],))

    # Create a network for the nonlinear response + LR
    xg = Dense(88, activation='relu')(inputg1)
    xg = Dense(46, activation='relu')(xg)
    z = Dense(5, activation='linear')(xg)

    # Define the model
    model = Model(inputs=inputg1, outputs=z)

    # Train
    model.compile(optimizer='rmsprop', loss=nll_gumbel)
    model.fit(X_train, Y_train, epochs=5, steps_per_epoch=10000)
    model.save("model_synthetic_beta")

    # Apply to data
    Y_pred = model.predict(X_test)

    # Plot the parity for each predicted output
    for n in range(0,5):
        fig, ax = plt.subplots(1,1)
        ax.errorbar(Y_test[:,n] + X_test[:,n], X_test[:,n], yerr=(Y_pred[:,n]**2 + 1),
                ls='', marker='o', mfc='none', mec='b')

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
Y = np.real(np.concatenate((np.log(stress_data[:,0:2]), 10*np.transpose(np.array([stress_data[:,2]])),
    np.array(Cv), np.array(Dv)), axis=1))

#Y = np.real(10*np.transpose(np.array([stress_data[:,2]])))

# Filter only the imaginary part (real part is very noisy)
J1 = np.imag(J1)
J3 = np.imag(J3)

# Take the absolute value
X1 = np.log(np.abs(J1))
X3 = np.log(np.abs(J3))

# Load trained model and predict
model_maps = keras.models.load_model("model_synthetic_mean")
Y_pred = model_maps.predict([X1,X3])
Y_diff = Y - Y_pred

# Split into training and testing
X_train = Y_pred[:9000,:]
X_test = Y_pred[9000:,:]
Y_train = Y_diff[:9000,:]
Y_test = Y_diff[9000:,:]

# Select task
sgr_maps_unc(matlib.repmat(X_train,10,1),matlib.repmat(Y_train,10,1),X_test,Y_test)

