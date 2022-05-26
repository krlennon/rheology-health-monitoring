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

## Contibuting

Inquiries and suggestions can be directed to krlennon[at]mit.edu.

## License

[GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)

