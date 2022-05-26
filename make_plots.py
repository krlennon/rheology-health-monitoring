import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.models
import pandas as pd
import timeit


# Get the models
model_maps = keras.models.load_model("model_maps")
model_lr = keras.models.load_model("model_lr")

# Get the MAPS data
X_test_maps = np.load("maps_data.npy")
J1_test = np.imag(X_test_maps[:,:3])
J3_test = np.imag(X_test_maps[:,3:])
X_test_1 = np.log(np.abs(J1_test))
X_test_3 = np.log(np.abs(J3_test))

# Get and plot the LR data
mins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
X_test_lr = np.zeros((len(mins),8))
fig1, ax1 = plt.subplots(1,1)
for i in range(len(mins)):
    df = pd.read_csv(f"data/saos/saos_{mins[i]}min.csv", header=None)
    data = df.to_numpy()
    w = np.flip(data[:,3])
    Gp = np.flip(data[:,0])
    Gpp = np.flip(data[:,1])
    X_test_lr[i,:] = Gpp
    ax1.loglog(w, Gpp, 'bo--', alpha=(i+2)/(len(mins)+1))
ax1.tick_params(which="both", direction="in", top=True, right=True, labelsize=12)
X_test_lr = np.log(np.abs(X_test_lr))

# Predict the noise temperature
x_lr = model_lr.predict(X_test_lr)[:,2]/10
start = timeit.timeit()
x_maps = model_maps.predict([X_test_1, X_test_3])[:,2]/10
stop = timeit.timeit()
print(stop - start)

# Plot predictions
fig2, ax2 = plt.subplots(1,1)
ax2.plot(mins, x_lr, 'ro')
ax2.plot(mins[:-2], x_maps, 'bo')
ax2.tick_params(which="both", direction="in", top=True, right=True, labelsize=12)
plt.show()

