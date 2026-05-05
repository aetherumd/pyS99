import numpy as np
from scipy.stats import norm
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
"""
Notes:
t0 is 566 Myr
"""

def s99_loader(dir_fil = '/Users/lamoreau/python/ASpec/Starburst99/output/salpeter.spectrum1'):
    n1 = 1
    n2 = 2
    NN = 150 #time bins
    dim = 1221 #number of wavelength bins
    nrows = dim*NN #total number of bins in a spectrum file
    time, wav, y = np.loadtxt(dir_fil, unpack=True, usecols=(0,n1,n2), max_rows=nrows, skiprows=6) #unpack seperates columns (how???, set " " as delimiter)

    time=time/1e6
    imax=len(y)/dim
    print(imax)

    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

    def Htimereal(z):
          return cosmo.age(z).to(u.Myr)

    def Htime(z): #hubble time
            t0=566.0
            return t0*((1+z)/10.)**-1.5
    
    def Hredshift(t): #hubble redshift
            t0=566.0
            return 10.0*(t/t0)**-(2./3)-1.0
    
    lums = np.array([3.90955858e+41, 3.31949205e+36, 2.26410702e+34, 2.72101305e+36,
    1.20475976e+37, 2.26032755e+37, 8.64213222e+35, 2.53087718e+36,
    9.26514983e+35, 8.30460881e+36, 1.03474205e+42, 4.13492301e+33,
    4.83716059e+35, 1.49603706e+37, 1.26528935e+37, 4.24149818e+29,
    2.19186638e+37, 1.10545778e+37, 5.37404641e+35, 1.63110992e+35,
    6.97985519e+32, 3.48213989e+32, 1.58560762e+33, 1.32286373e+36,
    2.66617959e+36])
    lines = np.array(['H1 6562.80A', 'O1 1304.86A', 'O1 6300.30A', 'O2 3728.80A',
        'O2 3726.10A', 'O3 1660.81A', 'O3 1666.15A', 'O3 4363.21A',
        'O3 4958.91A', 'O3 5006.84A', 'He2 1640.41A', 'C2 1335.66A',
        'C3 1906.68A', 'C3 1908.73A', 'C4 1549.00A', 'Mg2 2795.53A',
        'Mg2 2802.71A', 'Ne3 3868.76A', 'Ne3 3967.47A', 'N5 1238.82A',
        'N5 1242.80A', 'N4 1486.50A', 'N3 1749.67A', 'S2 6716.44A',
        'S2 6730.82A'])
    
    # ---------------------------------
    # Convert "O1 1304.86A" → "1304.86"
    # ---------------------------------
    lines = np.array([float(l[3:10]) for l in lines])
    print(type(lines), type(lums))
    diff_lum = lums/lines
    def gaussian_adder(xarr, means, areas, sigmas): #I should add error handling for this
        sigmas = np.atleast_1d(sigmas)
        yarr = np.zeros_like(xarr)
        if len(sigmas) == 1:
            for i in range(len(means)):
                yarr += areas[i] / (sigmas*np.sqrt(2*np.pi)) * np.exp(-((xarr - means[i])**2) / (2 * sigmas**2))
        elif len(sigmas) == len(means):
            for i in range(len(means)):
                yarr += areas[i] / (sigmas[i]*np.sqrt(2*np.pi)) * np.exp(-((xarr - means[i])**2) / (2 * sigmas[i]**2))
        else:
             print("Shape mismatch for gaussian addition function")
        return 
    print(len(lines), len(diff_lum))

    line_lum = gaussian_adder(wav[0:1220], lines, diff_lum, 150/2.355) #2.355 for FWHM to std_dev 

    
    # mass1=5e3
    #what is t0 for the sed_continuum code
    for ii in range(0,len(t0)-1): #this is taking the desired time stamp and pulling out the correct wavelength
        i=int(t0[ii]/2.0) #what is this divide by two
        if (i >=0 and i<imax):
            a1=dim*i
            b1=dim*(i+1)-1
            # weight=mass1[ii]/1.e6
            y_tot= 10**y[a1:b1] + line_lum
                #print(wav_obs[a1],wav_obs[b1])
    return wav, y_tot, a1, b1

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    wav, lum, a1, b1 = s99_loader()
    print(len(wav))
    print(wav)
    print(len(lum))
    print(lum)
