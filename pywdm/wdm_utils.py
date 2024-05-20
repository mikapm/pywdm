import numpy as np 
import xarray as xr 
from pywdm import wdm

"""
Functions to apply WDM on various datasets.
"""

def wdm_stereo(eta, xc=-5, yc=-90, sidelen=10, fs=5, local=True):
    """
    Get WDM directional wave spectrum from virtual array of wave staffs
    in stereo video grid within xr.Dataset 'ds'.
    
    Parameters:
        ds - xr.DataArray; space-time stereo grid
        xc - scalar; array origin x coordinate
        yc - scalar; array origin y coordinate
        sidelen - scalar; array half side length in m
        fs - scalar; sampling frequency in Hz
        local - bool; if True, uses local x,y coordinate system
                for spectrum
    Returns:
        dse - xr.Dataset; frequency-directional wave spectrum from WDM
        dsw - xr.Dataset; WDM output: amplitudes, frequencies & wavenumbers
    """
    # WDM analysis (center array at max crest grid cell)
    origin = (xc, yc) # x,y coordinates of array origin
    # x,y coordinates of square array for WDM
    xpts = [origin[0]+sidelen, origin[0], origin[0]-sidelen, origin[0]]
    ypts = [origin[1], origin[1]+sidelen, origin[1], origin[1]-sidelen]
    # Define A and R arrays for stereo array
    a_p1 = 0 # Angle from origin to p1 
    a_p2 = 90
    a_p3 = 180
    a_p4 = 270
    # Angles from center
    A = np.array([a_p1, a_p2, a_p3, a_p4]).astype(float)
    # Convert angles to radians
    A *= (np.pi / 180)
    # Radius of array (m). i.e. Polar coordinates of the staffs
    o_l1 = sidelen + 0 # Distance from origin to p1
    o_l2 = sidelen + 0
    o_l3 = sidelen + 0
    o_l4 = sidelen + 0
    R = np.array([o_l1, o_l2, o_l3, o_l4])
    # Make virtual wave staff array (interpolate over NaNs)
    tc = 'time'
    arr = np.array([eta.sel(x=xpts[0], y=ypts[0], method='nearest').interpolate_na(dim=tc).bfill(
                            dim=tc).ffill(dim=tc).values.squeeze(), 
                    eta.sel(x=xpts[1], y=ypts[1], method='nearest').interpolate_na(dim=tc).bfill(
                            dim=tc).ffill(dim=tc).values.squeeze(), 
                    eta.sel(x=xpts[2], y=ypts[2], method='nearest').interpolate_na(dim=tc).bfill(
                            dim=tc).ffill(dim=tc).values.squeeze(),
                    eta.sel(x=xpts[3], y=ypts[3], method='nearest').interpolate_na(dim=tc).bfill(
                            dim=tc).ffill(dim=tc).values.squeeze(),
                    ])
    # Check that array is even
    if not (arr.shape[-1] % 2) == 0:
        arr = arr[:,:-1]
    # Initialize WDM class using default parameters
    WDM = wdm.WaveletDirectionalMethod(A, R, lf=0.05, hf=0.8, nv=3, ns=fs)
    # Run WDM algorithm: get wavelet amp., wavenumbers, directions and frequencies
    Amp, K, Th, freqs = WDM.wdm_kth(arr.T)
    # Rotate, flip and sort directions
    theta = Th * (180/np.pi)
    # Make dataset of WDM output
    dsw = xr.Dataset(data_vars={'Amp':(['freq','time'],Amp), 
                                'K':(['freq','time'],K), 
                                'Th':(['freq','time'],theta)
                                }, 
                     coords={'freq': (['freq'], freqs),
                             'time': (['time'], np.arange(Amp.shape[1])/fs),
                             },
                     )
    # Get f-theta spectrum (local coordinate system)
    res = 10 # Dir. resolution
    if not local:
        dse = WDM.spec_fth(Amp, Th*(180/np.pi), freqs, res=res, local_dir=False)
    else:
        dse = WDM.spec_fth(Amp, Th*(180/np.pi), freqs, res=res, local_dir=True, ang_offset=21)
    
    return dse, dsw