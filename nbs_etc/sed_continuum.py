import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# Script creates SED, need to figure out how to add these

def sed_cont(dir_fil = '/Users/lamoreau/python/ASpec/Starburst99/output/salpeter.spectrum1'):
    xx=4
    # wav = amstrong, y1 = erg/s/Amstrong
    # y1= 3  Myr
    # y2= 5 Myr
    n1=1
    n2=2
    NN=150 #time bins
    #NN=400
    dim=1221 #number of wavelength bins
    nrows=dim*NN 
    #time,wav, y1 = np.loadtxt('output49291/Kroupa-lowmet1.spectrum1', unpack=True, usecols=(0,n1,n2),max_rows=nrows) 
    #time,wav, y1 = np.loadtxt('output15012/Kroupa.spectrum1', unpack=True, usecols=(0,n1,n2),max_rows=nrows)
    #time,wav, y1 = np.loadtxt('output61239/Kroupa-lowmet.spectrum1', unpack=True, usecols=(0,n1,n2),max_rows=nrows)
    #time,wav, y1 = np.loadtxt('output61239/Kroupa-lowmet.spectrum1', unpack=True, usecols=(0,n1,n2))
    #time,wav, y1 = np.loadtxt('output38744/top-heavy.spectrum1', unpack=True, usecols=(0,n1,n2),max_rows=nrows)
    #time,wav, y1b = np.loadtxt('output54345/top-heavy-lowmet.spectrum1', unpack=True, usecols=(0,n1,n2),max_rows=nrows)
    #time,wav, y1 = np.loadtxt('output/salpeter.spectrum1', unpack=True, usecols=(0,n1,n2),max_rows=nrows, skiprows = 6)
    time,wav, y1 = np.loadtxt(dir_fil, unpack=True, usecols=(0,n1,n2),max_rows=nrows, skiprows = 6)
    time=time/1e6
    imax=len(y1)/dim
    print(imax)

    def Htime(z): #hubble time
            t0=566.0
            return t0*((1+z)/10.)**-1.5

    def Hredshift(t): #hubble redshift
            t0=566.0
            return 10.0*(t/t0)**-(2./3)-1.0

    time0=590.0 #choose initial time
    #time0=595.0
    redsh0=Hredshift(time0) #calc initial redshift
    time1=530.0 #choose final time
    redsh1=Hredshift(time1) #calc final redshift
    s99dir = "/Users/lamoreau/python/ASpec/Starburst99/"
    #redshift, star forming complex radius (pc), mass star formation complex (Msun), number density of hydrogen (units of cm^-3), metallicity (Zsun)
    z1,ra,mass1,n_H,met=np.loadtxt(s99dir + 'logSFC-CCFid.txt',unpack=True, usecols=(2,4,5,8,9))
    z2,ra,mass2,n_H,met=np.loadtxt(s99dir + 'logSFC-low1.txt',unpack=True, usecols=(2,4,5,8,9))
    z3,ra,mass3,n_H,met=np.loadtxt(s99dir + 'logSFC-high1.txt',unpack=True, usecols=(2,4,5,8,9)) 
    Ht1=Htime(z1) #passes in a redshift array, get a time array out
    Ht2=Htime(z2)
    Ht3=Htime(z3)

    run='CC-Fid'
    #run='SFE 0.70'
    #run='SFE 0.35'
    z=z1
    Ht=Ht1
    mass=mass1

    t0=time0-Ht #get an array of star ages
    tmax0=np.max(t0)
    tmin0=np.min(t0)
    t1=time1-Ht #get an array of star ages
    tmax1=np.max(t1)
    tmin1=np.min(t1)
    print(time0,tmax0,tmin0)
    print(time1,tmax1,tmin1)

    #SFR
    tstart=300
    tend=700
    nbin=int((tend-tstart)/1.0) 
    print('nbins=',nbin)
    ti=np.linspace(300,700,nbin)
    dti=(ti[1:]-ti[:-1])*1e6 #change in ti? this is the timestep

    #dt=(Ht[1:]-Ht[:-1])*1e6
    #sfr=(massc[1:]-massc[:-1])/dt

    #getting the star formation rates
    massc=np.cumsum(mass)
    print('CC',massc[-2:-1],z[-2:-1])
    massci=np.interp(ti,Ht,massc)
    sfri=(massci[1:]-massci[:-1])/dti

    massc1=np.cumsum(mass1)
    print('CC',massc1[-2:-1],z1[-2:-1])
    massci1=np.interp(ti,Ht1,massc1)
    sfri1=(massci1[1:]-massci1[:-1])/dti

    massc2=np.cumsum(mass2)
    print('low',massc2[-2:-1])
    massci2=np.interp(ti,Ht2,massc2)
    sfri2=(massci2[1:]-massci2[:-1])/dti

    massc3=np.cumsum(mass3)
    print('high',massc3[-2:-1])
    massci3=np.interp(ti,Ht3,massc3)
    sfri3=(massci3[1:]-massci3[:-1])/dti


    def Halpha(QHI):
        #erg/s
        # xi_ion=QHI/Lv1500[s-1/(erg s-1 Hz-1] [Hz/erg]
        #typical Lv1500~1e27*(M_*/1e8Msun)
        #typical xi_ion=3e25
        #typical QHI=xi_ion*Lv1500=3e52 s-1(M_*/1e8Msun)
        return 1.36e-12*QHI

    def Lv1500(Muv):
        # log10 Lv1500 (erg s−1 Hz−1)
        return 0.4*(51.63-Muv)

    def Ll1500(Muv):
        #lam=c/nu
        #L_lam=(c/lam^2)dL/dnu
        # log10 Ll1500 (erg s−1 Ams-1)
        return Lv1500(Muv)+np.log10(3.e10/(1500.*1e-8)/1500.)

    Muv=-14
    # 1e6 Msun starbust Ll1500 = 39 - 39.5 the first few Myr
    print(f"Mag = {Muv:f} -> Solar masses (10Myr): {1e6*10**(Ll1500(Muv)-39):e}")
    #EW > 2080 A
    #EW ~3000 A for Salpeter IMF to 100 Msun t<5 Myr

    mag=120
    mass1=5e3

    #xf=mag*mass1/1e6 #some kind of luminosity per million solar masses
    xf=1.0

    #z=6.639
    # in cm (Ned's calc)
    #d_L=66348.7*3.08e24

    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    d_L0=cosmo.luminosity_distance(redsh0).value*3.08e24 # cm to Mpc
    d_L1=cosmo.luminosity_distance(redsh1).value*3.08e24 
    #print(d_L0,d_L)

    # in micron
    wav_obs0=wav*(redsh0+1)*1e-4
    wav_obs1=wav*(redsh1+1)*1e-4
    # F=L/(4pi*d_L^2) [erg/s/cm2/Amsr]
    # L_lamb/(1+z) L/(1+z)
    y1_obs0=np.array(xf*10**(y1-np.log10(4.*np.pi)-2*np.log10(d_L0))/(1+redsh0))
    y1_obs1=np.array(xf*10**(y1-np.log10(4.*np.pi)-2*np.log10(d_L1))/(1+redsh1))
    # convert to micro Jy: 1e-23 cgs

    # convert to uJy
    mag=10.0
    nu=3e10/(wav*1e-8)
    conv=wav/nu
    y1_obs0=y1_obs0*conv/1e-29*mag*(1+redsh0)**2
    y1_obs1=y1_obs1*conv/1e-29*mag*(1+redsh1)**2

    y_tot0=np.zeros(dim-1)
    y_tot1=np.zeros(dim-1)
    for ii in range(0,len(t0)-1):
            i=int(t0[ii]/2.0)
            if (i >=0 and i<imax):
                #print(ii,i)
                a1=dim*i
                b1=dim*(i+1)-1
                weight=mass[ii]/1.e6
                y_tot0=y_tot0+weight*y1_obs0[a1:b1]    
                #print(wav_obs[a1],wav_obs[b1])

    for ii in range(0,len(t1)-1):
            i=int(t1[ii]/2.0)
            if (i >=0 and i<imax):
                a1=dim*i
                b1=dim*(i+1)-1 #off by one issue
                weight=mass[ii]/1.e6
                y_tot1=y_tot1+weight*y1_obs1[a1:b1] 
    return wav_obs0, wav_obs1, a1, b1, y_tot0, y_tot1, run, time0, time1, redsh0, redsh1, ti, sfri, Hredshift, Htime, [z1, z2, z3], [massc1, massc2, massc3]



if __name__ == "__main__":
    wav_obs0, wav_obs1, a1, b1, y_tot0, y_tot1, run, time0, time1, redsh0, redsh1, ti, sfri, Hredshift, Htime, zs, masses = sed_cont()
    fig,axs = plt.subplots(2, 1, figsize=(6, 3.5), dpi=200)
    plt.subplots_adjust(hspace=0.85,top=0.95,right=0.95,bottom=0.13)
    axs[0].plot(wav_obs0[a1:b1], y_tot0, label=f'{run:s} t={time0:3.0f} Myr (z={redsh0:2.2f})')
    axs[0].plot(wav_obs1[a1:b1], y_tot1, label=f'{run:s} t={time1:3.0f} Myr (z={redsh1:2.2f})')
    axs[0].set_yscale('log')
    #axs[0].set_yscale('linear')
    axs[0].set_xlim(0.5, 6)
    #axs[0].set_ylim(1e-25,1e-20)
    axs[0].set_ylim(1e-5,1e-2)
    axs[0].set_xlabel(r"$\lambda_{obs}/\mu$m")
    #axs[0].set_ylabel(r"F$_\lambda(erg/s/cm^2/A)$")
    axs[0].set_ylabel(r"F$_\nu(\mu Jy)$")
    axs[0].legend(fontsize="6", loc ="lower right")

    axs[1].plot([time0,time0],[0,10],'--')
    axs[1].plot([time1,time1],[0,10],'--')

    #axs[1].plot(ti[1:],sfri1)
    #axs[1].plot(ti[1:],sfri2)
    #axs[1].plot(ti[1:],sfri3)
    axs[1].plot(ti[1:],sfri)

    axs[1].set_xlabel(r"time [Myr]")
    axs[1].set_ylabel(r"SFR [M$_\odot$/yr]")

    secax = axs[1].secondary_xaxis('top', functions=(Hredshift,Htime))
    secax.set_xlabel('Redshift')

    #fig1= plt.subplots(1, 1)
    #ax1.plot(age[1:]-age[:-1])
    #print(age[0],age[1],age[2])
    #plt.ylim(0.01,4)
    axs[1].set_xlim(350,700)
    axs[1].set_ylim(1e-3,2.0)
    plt.yscale('linear')
    plt.yscale('log')


    fig,ax1 = plt.subplots(1, 1)
    ax1.plot(zs[0],masses[0],label='CC')
    ax1.plot(zs[1],masses[1],label='low')
    ax1.plot(zs[2],masses[2],label='high')
    ax1.set_xlim(13,6)
    plt.legend()

    plt.show()