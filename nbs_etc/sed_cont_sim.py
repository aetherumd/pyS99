import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from sim_load import sim_load


def sed_cont_sim(ad):
    # TODO: Make all the functions dynamic
    dir_fil = '/Users/lamoreau/python/ASpec/Starburst99/'
    
    
    redsh0 = ds.current_redshift

    unit_t = ds.parameters["unit_t"]  # seconds
    time_now_Myr = ds.current_time.to("Myr").value * u.Myr  # 360.4 Myr

    # Birth epoch is dimensionless in yt, multiply by unit_t to get seconds
    star_birth_offset = ad["star", "particle_birth_epoch"].v * unit_t * u.s

    # Formation time (cosmic time when star was born)
    formation_time_Myr = time_now_Myr + star_birth_offset.to(u.Myr)

    #Retreiving code unit masses for particles (should all be ten)
    star_mass_code = ad["star", "particle_mass"]

    # Stellar ages and masses
    star_age_Myr = time_now_Myr - formation_time_Myr #t0 from Dr. Ricottis code
    star_mass = (star_mass_code * ds.mass_unit.to("Msun")).v * u.Msun #mass in dr ricottis code
    t0 = star_age_Myr.value
    mass = star_mass.value

    # lookup table params
    n1=1
    n2=2
    NN=150 #time bins
    #NN=400
    dim=1221 #number of wavelength bins
    nrows=dim*NN 
    
    # extracting from the lookup table
    time, wav, y1 = np.loadtxt(dir_fil + 'output/salpeter.spectrum1', unpack=True, usecols=(0,n1,n2),max_rows=nrows, skiprows = 6)

    #transformations on lookup table
    time=time/1e6 # get time into megayears
    imax=len(y1)/dim
    print(imax)

    def Htime(z): #hubble time
            t0=566.0
            return t0*((1+z)/10.)**-1.5

    def Hredshift(t): #hubble redshift
            t0=566.0
            return 10.0*(t/t0)**-(2./3)-1.0
    
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    d_L0=cosmo.luminosity_distance(redsh0).value*3.08e24 # cm to Mpc

    # in micron
    wav_obs0=wav*(redsh0+1)*1e-4

    # TODO: break down what this means/why have it
    xf=1.0
    y1_obs0=np.array(xf*10**(y1-np.log10(4.*np.pi)-2*np.log10(d_L0))/(1+redsh0))

    # convert to uJy
    mag=1.0 # was 10
    nu=3e10/(wav*1e-8)
    conv=wav/nu
    y1_obs0=y1_obs0*conv/1e-29*mag*(1+redsh0)**2

    y_tot0=np.zeros(dim-1)
    for ii in range(0,len(t0)-1): #this is taking the desired time stamp and pulling out the correct wavelength
            i=int(t0[ii]/2.0)
            if (i >=0 and i<imax):
                #print(ii,i)
                a1=dim*i
                b1=dim*(i+1)-1
                weight=mass[ii]/1.e6
                y_tot0=y_tot0+weight*y1_obs0[a1:b1]    
                #print(wav_obs[a1],wav_obs[b1])
    return wav_obs0, a1, b1, y_tot0, redsh0



if __name__ == "__main__":
    from Fitser_v2 import JWST_disperser as disp
    ds, wavelengths = sim_load()
    ad = ds.all_data()
    wav_obs0, a1, b1, y_tot0, redsh0 = sed_cont_sim(ad=ad)
    disp_name = "prism"
    prism = disp("prism")
    pcent, bflux = prism.binner(y_tot0, wav_obs0[a1:b1])
    fig,axs = plt.subplots(1, 1, figsize=(6, 3.5), dpi=200)
    plt.subplots_adjust(hspace=0.85,top=0.95,right=0.95,bottom=0.13)
    axs.plot(wav_obs0[a1:b1], y_tot0, label=f'(z={redsh0:2.2f})', lw = 1)
    axs.step(pcent, bflux, where = "mid", label = f"Binned Flux in {disp_name} disperser", color = "red", lw = 1)
    axs.set_yscale('log')
    axs.set_xlim(0.5, 8)
    axs.set_ylim(1e-10,1e-4)
    axs.set_xlabel(r"$\lambda_{obs}/\mu$m")
    axs.set_ylabel(r"F$_\nu(\mu Jy)$")
    axs.legend(fontsize="6", loc ="lower right")

    plt.yscale('linear')
    plt.yscale('log')

    plt.show()  