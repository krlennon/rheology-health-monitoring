import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.models
import pandas as pd
import timeit

def nll_gumbel(y_true, y_pred):
    # Negative log-likelihood for a Gumbel distribution
    z = tf.divide(y_true, tf.square(y_pred))
    nll = z + tf.math.exp(tf.math.scalar_mul(-1, z)) + tf.math.log(tf.square(y_pred))
    return tf.reduce_mean(nll, axis=-1)

def gumbel(x, mu, beta):
    # The Gumbel distribution
    z = (x - mu)/beta
    return np.exp(-z - np.exp(-z))/beta

def get_gumbel_quartiles(mus, betas):
    # Get the lower and upper quartile for the Gumbel distribution
    lowers = []
    uppers = []
    for i in range(len(mus)):
        lowers += [-betas[i]*np.log(-np.log(0.25)) + mus[i]]
        uppers += [-betas[i]*np.log(-np.log(0.75)) + mus[i]]
    return np.array(lowers), np.array(uppers)

# Get the models
model_maps = keras.models.load_model("model_maps")
model_maps_unc = keras.models.load_model("model_synthetic_beta", custom_objects={"nll_gumbel":nll_gumbel})
model_lr = keras.models.load_model("model_lr")
model_lr_beta = keras.models.load_model("model_synthetic_beta_lr", custom_objects={"nll_gumbel":nll_gumbel})

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

# Predict the noise temperature and uncertainty
Y_lr = model_lr.predict(X_test_lr)
x_lr = Y_lr[:,2]/10
x_beta_lr = (model_lr_beta.predict(Y_lr)[:,2]**2)/10
start = timeit.timeit()
Y_maps = model_maps.predict([X_test_1, X_test_3])
x_maps = Y_maps[:,2]/10
x_beta_maps = (model_maps_unc.predict(Y_maps)[:,2]**2 + 1)/10
stop = timeit.timeit()
print(stop - start)

# Make get quartiles
lowers_lr, uppers_lr = get_gumbel_quartiles(x_lr, x_beta_lr)
lowers_maps, uppers_maps = get_gumbel_quartiles(x_maps, x_beta_maps)

# Plot predictions
fig2, ax2 = plt.subplots(1,1)
ax2.errorbar(mins, (uppers_lr+lowers_lr)/2, yerr=(uppers_lr-lowers_lr)/2, marker='', ls='', color='r', zorder=1)
ax2.errorbar(mins[:-2], (uppers_maps+lowers_maps)/2, yerr=(uppers_maps-lowers_maps)/2,
        marker='', ls='', color='b', zorder=2)
ax2.scatter(mins, x_lr, marker='o', color='r', zorder=3)
ax2.scatter(mins[:-2], x_maps, marker='o', color='b', zorder=3)
ax2.set_ylim([1,7])
ax2.tick_params(which="both", direction="in", top=True, right=True, labelsize=12)
plt.savefig("noise_temperature.pdf", transparent=True)

# Make plots of distributions at two times
fig3, ax3 = plt.subplots(1,1)
x = np.linspace(0,10,1000)
ax3.plot(x, gumbel(x,x_lr[5],x_beta_lr[5]),'r--')
ax3.plot(x, gumbel(x,x_maps[5],x_beta_maps[5]),'b--')
ax3.plot(x, gumbel(x,x_lr[7],x_beta_lr[7]),'r')
ax3.plot(x, gumbel(x,x_maps[7],x_beta_maps[7]),'b')
ax3.set_xlim([1,7])
ax3.set_ylim([0,2])
ax3.tick_params(which="both", direction="in", top=True, right=True, labelsize=12)

plt.show()

