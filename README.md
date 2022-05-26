# Rheology Health Monitoring

This repository contains the data and code for the paper "Anticipating gelation and vitrification with medium amplitude parallel superposition (MAPS) rheology and artificial neural networks"..

Publication of this work is forthcoming. For now, if you use the data or code within this repository, please cite it using the metadata in the [citation](CITATION.cff) file.

## Contents

### `make_plots.py`

Short script for making two figures: a plot of the time-resolved loss modulus from SAOS frequency sweeps, and a plot of the time-resolved noise temperature computed from pre-trained ANNs ([linear response](model_lr) and [MAPS response](model_maps)) using [SAOS](data/saos) and [MAPS](data/maps) data.

### `maps_data.npy`

Preprocessed MAPS data, containing the values of the first and third order commplex compliances. Computed from [raw MAPS data](data/maps) using `process_maps.py`.

### `process_maps.py`

Short script to process [raw MAPS data](data/maps). 

### `sgr_lr.py`

Script for training an ANN to compute the parameters of the SGR model (including the noise temperature) from SAOS frequency sweep data. The ANN is trained on [synthetic SAOS data](data/synthetic/SGR_LR.mat). A pre-trained ANN is found [here](model_lr).

### `sgr_maps.py`

Script for training an ANN to compute the parameters of the SGR model (including the noise temperature) from a single MAPS experiment. The ANN is trained on [synthetic MAPS data](data/synthetic/SGR_data_tensorial_569.mat). A pre-trained ANN is found [here](model_maps).

### `sgr_synthetic.py`

Script for training a set of ANNs to compute the parameters of the SGR model from synthetic MAPS data (with tone sets [(5,6,9)](data/synthetic/SGR_data_tensorial_569.mat) and [(1,4,16)](data/synthetic/SGR_data_tensorial_1416.mat)).

### `data`

Contains all of the raw data and scripts to generate synthetic data (see the subdirectory [README](data/README.md)).

### `model_lr`

Trained Keras model for determining SGR parameters from SAOS data.

### `model_maps`

Trained Keras model for determining SGR parameters from MAPS data.

## Contibuting

Inquiries and suggestions can be directed to krlennon[at]mit.edu.

## License

[GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)

