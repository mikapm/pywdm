# pywdm
Python implementation of the Wavelet Directional Method (WDM) for estimating directional wavefield properties from arrays of wave probes as described by [Donelan et al. (1996; J. Phys. Oceanogr.)](https://doi.org/10.1175/1520-0485(1996)026<1901:NAOTDP>2.0.CO;2). This implementation is a more or less direct translation of the original WDM Matlab codes created by Mark Donelan.

The WDM functions are found in the wdm.py file, and example jupyter notebook scripts are provided in ekok_stereo.ipynb (Ekofisk stereo video data) and wdm_tests.ipynb (Ekofisk laser array data).

## Installation
```
$ mamba env create -f environment.yml
$ conda activate wdm
```

## Update environment
```
mamba env update --file environment.yml --prune
```
