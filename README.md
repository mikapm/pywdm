# pywdm
Python implementation of the Wavelet Directional Method (WDM) for estimating directional wavefield properties from arrays of wave probes as described by [Donelan et al. (1996; J. Phys. Oceanogr.)](https://doi.org/10.1175/1520-0485(1996)026<1901:NAOTDP>2.0.CO;2). This implementation is a more or less direct translation of the original WDM Matlab codes created by Mark Donelan.

The WDM functions are found in the wdm.py file, and an example jupyter notebook script is provided in wdm_tests.ipynb. The example script uses data from an array of four laser altimeters on the Ekofisk field in the central North Sea.

## Installation
```
$ mamba env create -f environment.yml
$ conda activate wdm
```

## Update environment
```
mamba env update --file environment.yml --prune
```
