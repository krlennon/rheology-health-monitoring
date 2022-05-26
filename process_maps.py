import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_all = []
for k in range(1,9):
    w0 = 0.2 # Fundamental frequency

    # Loop through amplitudes
    ws = []
    stressFTs = []
    strainFTs = []
    for amp in ["005", "01"]:
        # Read data
        df = pd.read_csv(f"data/maps/maps_{amp}Pa_2rads_{k}.csv", header=None)
        data = df.to_numpy()
        t = data[:,3]
        strain = data[:,4]
        stress = data[:,5]

        # Window
        t_begin = t[-1] - 4*(2*np.pi/w0)
        I = np.argmin(np.abs(t - t_begin))
        t = t[I:]
        strain = strain[I:]
        stress = stress[I:]

        # FFT
        ws += [2*np.pi*np.fft.rfftfreq(len(t), (t[-1] - t[0])/(len(t) - 1))]
        stressFTs += [2*np.pi*np.fft.rfft(stress)/len(t)]
        strainFTs += [2*np.pi*np.fft.rfft(strain)/len(t)]

    # Gets LR and MAPS
    n1, n2, n3 = (5, 6, 9)
    channels = [[n1, n2, n3], [n1, n2, -n3], [n1, -n2, n3], [n1, -n2, -n3], [n1, n1, n2], [n1, n1, -n2],
            [n1, n1, n3], [n1, n1, -n3], [n2, n2, n1], [n2, n2, -n1], [n2, n2, n3], [n2, n2, -n3], [n3, n3, n1],
            [n3, n3, -n1], [n3, n3, n2], [n3, n3, -n2], [n1, n1, n1], [n2, n2, n2], [n3, n3, n3]];

    # Find the linear response on the input channels
    J1 = []
    for n in [n1, n2, n3]:
        V = []
        b = []
        for i in range(2):
            # Build linear system
            I_w = np.argmin(np.abs(ws[i] - n*w0))
            stress_amp = stressFTs[i][I_w]
            strain_amp = strainFTs[i][I_w]
            V += [[1, stress_amp**2]]
            b += [strain_amp/stress_amp]

        # Solve the linear system
        V = np.array(V)
        b = np.array(b)
        x = np.linalg.lstsq(V, b)[0]
        J1 += [x[0]]

    # Find the MAPS response
    J3 = []
    for channel in channels:
        w_ch = w0*(channel[0] + channel[1] + channel[2])
        h_sym = False

        # Apply Hermitian symmetry if needed
        if w_ch < 0:
            w_ch = -w_ch
            h_sym = True

        # Build the linear system
        V = []
        b = []
        for i in range(2):
            I_w = np.argmin(np.abs(ws[i] - w_ch))
            strain_amp = strainFTs[i][I_w]
            if h_sym:
                strain_amp = np.conjugate(strain_amp)
            b += [strain_amp*(2*np.pi)**2]

            # Get the input amplitudes
            stress_prod = 1 + 0j
            stress_sum = 0j
            for n in channel:
                h_sym = False
                if n < 0:
                    n = -n
                    h_sym = True
                I_w = np.argmin(np.abs(ws[i] - n*w0))
                stress_amp = stressFTs[i][I_w]
                if h_sym:
                    stress_amp = np.conjugate(stress_amp)
                stress_prod *= stress_amp
                stress_sum += stress_amp
            V += [[stress_sum, stress_prod]]

        # Solve the linear system
        V = np.array(V)
        b = np.array(b)
        x = np.linalg.lstsq(V, b)[0]
        J3 += [x[1]]

    data_t = J1 + J3
    data_all += [data_t]

# Save data
data_all = np.array(data_all)
np.save("maps_data", data_all)

