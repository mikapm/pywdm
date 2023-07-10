#!/usr/bin/env python3

# Imports
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ************************************************************************
# General Wavelet Transform functions
def oct2freq(omin, omax, nv):
    """
    Compute wavelet transform (WT) frequencies from omin, omax, nv.
    (See wavelet() function for explanations of the parameters.)
    Based on getfreqs() by Daniel Santiago (https://github.com/dspelaez/wdm).
    """
    return 2.**np.linspace(omin, omax, nv * abs(omin-omax) + 1)

def morlet(s, nt, fs):
    """
    Function that generates Morlet wavelet. Based on momo.m function by Mark Donelan.
    
    Parameters:
        s - time (1/freq) (?)
        nt - length of timeseries to transform (nt = number of timesteps)
        fs - sampling frequency (Hz)
    """
    nu  = s * fs * np.arange(1, nt/2+1) / nt
    return np.exp(-1./np.sqrt(2) * ((nu - 1)/0.220636)**2.)

def wavelet(z, freqs, fs):
    """
    Wavelet transform function based on wavelet.m function by Mark Donelan, which
    in turn is based on Bertrand Chapron's Wavelet Transform code. Also inspired by
    the cwt_bc() function by Daniel Santiago (https://github.com/dspelaez/wdm).
    
    Parameters:
        z - 1D time series signal
        freqs - array of WT frequencies returned by oct2freq()
        fs - scalar; sampling frequency (Hz)
    """
    nt = len(z) # Time series length
    nf = len(freqs) # Frequency array length
    
    # Compute Morlet wavelet window function win
    win = np.zeros((nf, int(nt/2)))
    for i in range(nf):
        win[i,:] = morlet(1./freqs[i], nt=nt, fs=fs)
        
    # Take real fourier transform of signal
    fft = np.fft.fft(z)
    fft = fft[1:int(nt/2)+1] # 2nd half of FFT coefficients are redundant
    fft[0] /= 2
    
    # Compute wavelet coefficients (shape:[nf, nt]) by convolving the window 
    # and transformed time series
    return np.fft.ifft(2. * fft[None,:] * win, nt)
    

# ***************************************************************************


class WaveletDirectionalMethod():
    """
    Main WDM class. The functions in this class are mainly based on prior 
    Matlab codes from Mark Donelan and Jan-Victor Björkqvist as well as the
    Python/Fortran wdm algorithm by Daniel Santiago available at
    https://github.com/dspelaez/wdm.
    """
    
    def __init__(self, A, R, d=70, ang_offset=21, lf=0.05, hf=0.5, nv=3, ns=5):
        """
        Parameters:
            A - array; angles (rad) of wave staffs from array center
            R - array; radii (m) of staffs in array
            d - scalar; water depth
            ang_offset - scalar; c.c. offset (deg) of array orientation from due East
            lf - scalar; low-frequency cutoff
            hf - scalar; high-frequency cutoff
            nv - int; number of voices into which each octave is broken (number of points 
                      between each order of magnitude)
            ns - scalar; sampling frequency (Hz)
        
        Default parameters are for Ekofisk laser array (LASAR).
        """
        # User-defined parameters
        self.A = np.array(A)
        self.R = np.array(R)
        self.depth = d
        self.ang_offset = ang_offset
        self.lf = lf
        self.hf = hf
        self.nv = nv
        self.ns = ns
        # Determine highest and lowest frequencies used in the wavelet method (octaves),
        # such that fmin = 2**omin, fmax = 2**omax
        self.omin = int(np.floor(np.log(self.lf) / np.log(2))) # Min octave (lp in MD's code)
        self.omax = int(np.ceil(np.log(self.hf) / np.log(2))) # Max octave (hp in MD's code)
        # Other parameters
        self.nstaffs = len(A) # Number of wave staffs (np in MD's code)
        # Number of pairs (npp in MD's code)
        self.npairs = int(self.nstaffs * (self.nstaffs - 1) / 2) 
        
        
    def separation_vectors(self):
        """
        Returns the separation vectors r_ij (r) and alpha_ij (a) in the notation of 
        Donelan et al. (2015). See e.g., eq. 12 in D2015 or eq. 7 in D1996. 
        """
        # Calculate the distances and angles between all pairs of wave staffs
        X = self.R * np.cos(self.A)
        Y = self.R * np.sin(self.A)
        # Compute the possible separations (x_k - x_j), (y_k - y_j)
        # (see e.g. Kahma et al. 2005, p. 73)
        ij = 0 # Pair counter
        x = np.zeros(self.npairs)
        y = np.zeros(self.npairs)
        for j in range(self.nstaffs-1):
            for k in range(j+1, self.nstaffs):
                x[ij] = X[k] - X[j]
                y[ij] = Y[k] - Y[j]
                ij += 1
#                 x = np.array([(xk - xj) for xj in X[:-1] for xk in X[1:]])
#                 y = np.array([(yk - yj) for yj in Y[:-1] for yk in Y[1:]])  
                
        # Polar coords
        r = np.abs(x + 1j*y) # Complex coordinates
        a = np.arctan2(y, x) # Angles
        return r, a
    
    
    def calc_sv_diffs(self, r, a):
        """
        Calculate differences between possible pairs of wave staff separation vectors. 
        Input r and a returned by self.separation_vectors(). 
        
        Calculates the differences svd=(alpha_ab-alpha_cd) in the notation of D2015. 
        These differences were in the variable rr in the original code by MD.
        
        Only wave staff pairs whose svd is close to 90 or 270 deg are used. The indices
        for these pairs were stored in the variable ij in the original MD code.
        
        This functions returns the dictionary kth_coeffs containing the sin & cos terms 
        as well as r_ab and r_cd used in Eqs 8 and 9 in D1996. Also returns the indices
        for svd close to 90 or 270 deg for use in self.wdm().
        """
        # Iterate over possible pairs ab & cd and get separation vector differences 
        # (alpha_ab - alpha_cd) in the notation of D2015.
        svd = [] # Separation vector differences for each possible pair
        # Also initialize lists for r_ab&r_cd and sin&cos terms in eqs 8 & 9 in D1996
        sin_ab = []; sin_cd = []; cos_ab = []; cos_cd = []; r_ab = []; r_cd = []
        for ab in range(self.npairs-1):
            for cd in range(ab+1, self.npairs):
                diff = a[ab] - a[cd] # (alpha_ab - alpha_cd)
                # Convert negative angles to positive in the range 0-360
                rr = np.rad2deg(diff) # convert to angles
                rr %= 360 # correct negative angles
                svd.append(rr)
                # Compute some coefficients (cos & sin terms and r_ab & r_cd) to 
                # eqs 8 and 9 (k and theta equations) in D1996
                sin_ab.append(np.sin(a[ab]))
                sin_cd.append(np.sin(a[cd]))
                cos_ab.append(np.cos(a[ab]))
                cos_cd.append(np.cos(a[cd]))
                r_ab.append(r[ab])
                r_cd.append(r[cd])

        svd = np.array(svd) # To numpy array
        # Get indices of wave staff pairs whose svd is close to 90 or 270 deg.
        i90 = np.logical_and(svd>70, svd<110)
        i270 = np.logical_and(svd>250, svd<290)
        # Inds of svd close to 90 or 270 deg
        svd_inds = np.atleast_1d(np.argwhere(np.logical_or(i90, i270)).squeeze())

        # Save coefficients for eqs 8 and 9 in D1996
        kth_coeffs = {} # Empty dict
        kth_coeffs['sin_ab'] = np.array(sin_ab)
        kth_coeffs['sin_cd'] = np.array(sin_cd)
        kth_coeffs['cos_ab'] = np.array(cos_ab)
        kth_coeffs['cos_cd'] = np.array(cos_cd)
        kth_coeffs['r_ab'] = np.array(r_ab)
        kth_coeffs['r_cd'] = np.array(r_cd)

        return kth_coeffs, svd_inds
    

    def wavelet_arr(self, arr):
        """
        Compute continuous wavelet transform (WT) for array of time series arr 
        with shape [nt, nstaffs].
        Partly based on the wavelet_spectrogram() function by Daniel Santiago 
        available at https://github.com/dspelaez/wdm.
        
        Returns the wavelet spectrogram W (shape:[nfreq, ntime, nstaffs]), phase
        angles phi, wavelet mean amplitude Amp, and WT frequencies freq.
        """
        # Get WT frequencies
#         freqs = oct2freq(self.omin, self.omax, self.ns) # Gives different result from MD
        k = 0 # Counter
        freqs = [] # List for frequencies
        for i in np.arange(self.omin, self.omax+1):
            for j in range(self.nv):
                freqs.append(2**(i+j/self.nv))
        freqs = np.array(freqs)
        
        #  Compute wavelet coeffs, amplitudes and phase angles for each wave staff
        nt = len(arr) 
        nf = len(freqs)
        W = np.zeros((nf, nt, self.nstaffs), dtype='complex') # wavelet coeffs.
        Phi = np.zeros((nf, nt, self.nstaffs)) # phase angles
        Amp = np.zeros((nf, nt)) # wavelet amplitudes
        for i in range(self.nstaffs):
            wx = wavelet(arr[:,i], freqs, self.ns) # * np.sqrt(1.03565 / self.nv)
            W[:,:,i] = wx # Wavelet coefficients
            Phi[:,:,i] = np.angle(wx) # Phase angles
            Amp[:,:] += abs(wx) # Wavelet amplitudes
        # Take mean of amplitudes
        Amp /= self.nstaffs

        return W, Amp, Phi, freqs     
    
    
    def wdm_kth(self, arr):
        """
        Main WDM function for determining wavelet amplitude Amp, frequencies freqs,
        and wavenumber vector components K, Th from wave staff array arr using the
        WDM method of Donelan et al. (1996).
        
        Input array arr of wave staffs must be of shape (ntime, nstaffs).
        """
        # Wavelet transform of wave staffs
        W, Amp, phi, freqs = self.wavelet_arr(arr)
        
        # Separation vectors and coeffs for eqs 8 and 9 in D1996
        r, a = self.separation_vectors()
        kth_coeffs, svd_inds = self.calc_sv_diffs(r=r, a=a)
        npairs_to_use = len(svd_inds) # No. of pairs w/ separation vectors at 90 or 270 deg
        
        # Iterate over WT frequencies, compute wavenumbers k and directions th
        nf = len(freqs) # Number of WT frequencies
        nt = max(arr.shape) # Length of time series in array
        Dphi = np.zeros((nf, nt, self.npairs)) # All phase differences b/w pairs
        K = np.zeros((nf, nt, npairs_to_use)) # Wavenumbers
        Th = np.zeros_like(K) # Directions
        # n - iterator for frequencies
        # ij - iterator for pairs
        # i,j - iterators for staffs (j,k in MD's code)
        for n in range(nf):
            # Calculate phase differences pd between each pair of wave staffs for 
            # each WT frequency. These values were stored in variable b (shape:(j,k))
            # in original MD code.
            ij = 0 # Initialize pair counter
            for i in range(self.nstaffs-1):
                for j in range(i+1, self.nstaffs):
                    # Phase differences between pairs
                    Dphi[n,:,ij] = phi[n,:,j] - phi[n,:,i]
                    ij += 1 # Increase counter
            # Set phases in range (-pi, pi)
            Dphi[Dphi > np.pi] -= 2*np.pi
            Dphi[Dphi < -np.pi] += 2*np.pi

            # From JVB's version of MD's Matlab code (9.3.2018):
            # If Dphi=0 then Gamma will become Inf and we will get a direction of 45
            # instead of 90/270 degrees. This matters in generated test cases. 
            eps = 2.2204e-16
            Dphi[Dphi < eps] += eps
            
            # Calculate Gamma and phi_ab, phi_cd for eqs 8 and 9 in D1996
            ncombs = int(self.npairs * (self.npairs - 1) / 2) # Number of pair combinations
            ij = 0 # Initialize pair counter
            G = np.zeros((nt, ncombs)) # Gamma param. in eq. 9 of D1996
            r_ab = kth_coeffs['r_ab'] # r_ab in Gamma
            r_cd = kth_coeffs['r_cd'] # r_cd in Gamma
            phi_ab = np.zeros((nt, ncombs)) # Initialize phi_ab in D1996
            phi_cd = np.zeros_like(phi_ab)
            for i in range(self.npairs-1):
                for j in range(i+1, self.npairs):
                    phi_ab[:,ij] = Dphi[n,:,i] # phi_ab in eqs 8 & 9 of D1996
                    phi_cd[:,ij] = Dphi[n,:,j] # phi_cd in eqs 8 & 9 of D1996
                    G[:,ij] = (phi_ab[:,ij] / phi_cd[:,ij]) * (r_cd[ij] / r_ab[ij])
                    ij +=1 # Increase pair counter
            
            # Calculate the wavenumber k and direction th following eqs 8 and 9 in D1996
            # Only use pairs of wave staffs with separation vector near 90 or 270 deg; 
            # these indices are stored in svd_inds.
            kk_temp = np.zeros((nt, npairs_to_use)) # Temp. array
            th_temp = np.zeros_like(kk_temp) # Temp. array
            sin_ab = kth_coeffs['sin_ab'] # sin(alpha_ab)
            sin_cd = kth_coeffs['sin_cd'] # sin(alpha_cd)
            cos_ab = kth_coeffs['cos_ab'] # cos(alpha_ab)
            cos_cd = kth_coeffs['cos_cd'] # cos(alpha_cd)
            for i, l in enumerate(svd_inds):
                # Eq. (9) in Donelan et al. (1996)
                num_9 = (G[:,l] * cos_cd[l] - cos_ab[l]) # numerator
                den_9 = (sin_ab[l] - G[:,l]*sin_cd[l]) # denominator
                th_temp[:,i] = np.arctan2(num_9, den_9)
                # Eq. (8) in Donelan et al. (1996); use theta from previous eqn
                num_8 = ((phi_ab[:,l]/r_ab[l])*sin_cd[l] - (phi_cd[:,l]/r_cd[l])*sin_ab[l])
                den_8 = (cos_ab[l]*sin_cd[l] - cos_cd[l]*sin_ab[l]) / np.cos(th_temp[:,i])
                kk_temp[:,i] = num_8 / den_8
            # Correct directions for negative wavenumbers
            th_temp += (kk_temp<0) * np.pi
            kk_temp = abs(kk_temp)
            
            # Save to wavenumber and direction final arrays
            K[n,:,:] = kk_temp
            Th[n,:,:] = th_temp
        
        # Average K and Th if needed (i.e. if more than 3 staffs -> more than 1 pair)
        if self.nstaffs > 3:
            # For more staffs we get several K and Th estimates. In this case we save
            # the mean wavenumber and mean direction
            K = K.mean(axis=(2))
            Th = np.arctan2(np.mean(np.sin(Th), axis=(2)), np.mean(np.cos(Th), axis=(2)))
            
        return Amp, K.squeeze(), Th.squeeze(), freqs
    
    
    def spec_fth(self, Amp, Th, freqs, res=1):
        """
        Get frequency-direction (f-theta) spectrum from WDM analysis following MD's
        function direction_frequency.m
        
        Parameters:
            Amp - wavelet amplitudes (nf x nt)
            Th - wavelet directions in degrees (nf x nt)
            freqs - wavelet frequencies (nf)
            res - int; directional resolution in deg
        """
        # Number of frequencies and timesteps and directions
        nf, nt = Th.shape
        nd = int(360/res)
        
        # Calculate frequencies for normalizing (df) and initialize spectrum array Efd
        df = freqs * np.log(2) / self.nv # For normalizing the spectrum
        Efd = np.zeros((nf, nd)) # Spectrum array
        dth = res * np.pi/180 # Directional resolution in radians
        C = 1.03565; # This corrects for the non-orthogonality of the Morlet wavelets?
        
        # Populate spectrum
        dirs = np.arange(0, 360, int(res))
        for n in range(nf):
            for ct, d in enumerate(dirs):
                # Find array indices corresponding to directional bin
                ii = np.where(np.logical_and(Th[n,:]>d, Th[n,:]<(d+res)))
                # Amp has units m, and Efd will have units of m^2
                # Normalization according to code by Donelan
                Efd[n,ct] = np.sum(Amp[n,ii]**2) / self.nv * C / nt
        
        # Convert to units m^2/rad
        Efd /= dth
        # Convert to units m^2/rad/Hz
        Efd /= (df.reshape([-1, 1]) * np.ones(nd))
        # Frequency spectrum w/ units m^2/Hz
        S = np.sum(Efd, axis=1) * dth 
        
        return Efd, S


