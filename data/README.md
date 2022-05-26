# data

Directory containing all data and data generating scripts.

## Contents

### `maps`

Directory containing two files with raw MAPS data, both in stress control with a tone set of (5,6,9). One [file](maps/maps_005Pa_02rads_5min.txt) contains data for an amplitude of 0.05 Pa, and the [other](maps/maps_01Pa_02rads_5min.txt) contains data for an amplitude of 0.1 Pa. Both experiments used a fundamental frequency of 0.2 rad/s, and both collect data every five minutes.

### `saos`

Directory containing a series of files with SAOS data, collected at 0.1 Pa in stress control. Data was collected every 5 minutes using a frequency sweep from 10 rad/s to 0.5 rad/s, with a total of eight frequencies.

### `synthetic`

Directory containing synthetic [SOAS](synthetic/SGR_LR.mat) and MAPS ([(5,6,9)](synthetic/SGR_data_tensorial_569.mat) and [(1,4,16)](synthetic/SGR_data_tensorial_1416.mat)) data from the SGR model. Also contains a MATLAB scripts for [computing](synthetic/g3_sgr.m) the response of the SGR model, storing the data for the [MAPS ANN](synthetic/sgr_ann.m), and storing the data for the [SAOS ANN](synthetic/sgr_lr.m).
