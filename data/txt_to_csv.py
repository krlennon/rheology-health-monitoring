import numpy as np
import pandas as pd

# Read file
minutes = 5
with open(f"maps/maps_005Pa_02rads_{minutes}min.txt") as f:
    text = f.read()

text = text.split("[step]")
exps = []
strains = []
for exp in text[1:]:
    lines = exp.split("\n")
    data = []
    for line in lines[4:-2]:
        data_text = line.split()
        data_num = [float(d) for d in data_text]
        data += [data_num]
    data = np.array(data)
    exps += [data]
    strains += [np.max(data[:,2])/100]

for k in range(len(strains)):
    pd.DataFrame(exps[k]).to_csv(f"maps/maps_005Pa_02rads_{k+1}.csv", header=False, index=False)
