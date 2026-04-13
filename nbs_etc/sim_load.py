import merlin_spectra
from merlin_spectra.emission import EmissionLineInterpolator

import os
import copy

import numpy as np
from numpy import ndarray
import yt
from yt.frontends.ramses.field_handlers import RTFieldFileHandler

import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# Path to the installed package
merlin_path = os.path.dirname(merlin_spectra.__file__)

# Path to the linelists folder inside MERLIN
line_list = os.path.join(merlin_path, "linelists/linelist.dat")

print("MERLIN path:", merlin_path)
print("Line list path:", line_list)

filename = "/Users/lamoreau/python/ASpec/SimulationFiles/output_00273/info_00273.txt"
# need to make this a relative path that is consistant
# should be able to select files with GUI if we are going to have one

lines=["H1_6562.80A","O1_1304.86A","O1_6300.30A","O2_3728.80A","O2_3726.10A",
       "O3_1660.81A","O3_1666.15A","O3_4363.21A","O3_4958.91A","O3_5006.84A", 
       "He2_1640.41A","C2_1335.66A","C3_1906.68A","C3_1908.73A","C4_1549.00A",
       "Mg2_2795.53A","Mg2_2802.71A","Ne3_3868.76A","Ne3_3967.47A",
       "N5_1238.82A",
       "N5_1242.80A","N4_1486.50A","N3_1749.67A","S2_6716.44A","S2_6730.82A"]

wavelengths=np.array([6562.80, 1304.86, 6300.30, 3728.80, 3726.10, 1660.81, 1666.15,
             4363.21, 4958.91, 5006.84, 1640.41, 1335.66,
             1906.68, 1908.73, 1549.00, 2795.53, 2802.71, 3868.76,
             3967.47, 1238.82, 1242.80, 1486.50, 1749.67, 6716.44, 6730.82])

cell_fields = [
    "Density",
    "x-velocity",
    "y-velocity",
    "z-velocity",
    "Pressure",
    "Metallicity",
    "xHI",
    "xHII",
    "xHeII",
    "xHeIII",
]

epf = [
    ("particle_family", "b"),
    ("particle_tag", "b"),
    ("particle_birth_epoch", "d"),
    ("particle_metallicity", "d"),
]

# Ionization Parameter Field
# Based on photon densities in bins 2-4
# Don't include bin 1 -> Lyman Werner non-ionizing
def _ion_param(field, data):
    p = RTFieldFileHandler.get_rt_parameters(ds).copy()
    p.update(ds.parameters)

    cgs_c = 2.99792458e10     #light velocity

    # Convert to physical photon number density in cm^-3
    pd_2 = data['ramses-rt','Photon_density_2']*p["unit_pf"]/cgs_c
    pd_3 = data['ramses-rt','Photon_density_3']*p["unit_pf"]/cgs_c
    pd_4 = data['ramses-rt','Photon_density_4']*p["unit_pf"]/cgs_c

    photon = pd_2 + pd_3 + pd_4

    return photon/data['gas', 'number_density']


def _my_temperature(field, data):
    #y(i): abundance per hydrogen atom
    XH_RAMSES=0.76 #defined by RAMSES in cooling_module.f90
    YHE_RAMSES=0.24 #defined by RAMSES in cooling_module.f90
    mH_RAMSES=yt.YTArray(1.6600000e-24,"g") #defined by RAMSES in cooling_module.f90
    kB_RAMSES=yt.YTArray(1.3806200e-16,"erg/K") #defined by RAMSES in cooling_module.f90

    dn=data["ramses","Density"].in_cgs()
    pr=data["ramses","Pressure"].in_cgs()
    yHI=data["ramses","xHI"]
    yHII=data["ramses","xHII"]
    yHe = YHE_RAMSES*0.25/XH_RAMSES
    yHeII=data["ramses","xHeII"]*yHe
    yHeIII=data["ramses","xHeIII"]*yHe
    yH2=1.-yHI-yHII
    yel=yHII+yHeII+2*yHeIII
    mu=(yHI+yHII+2.*yH2 + 4.*yHe) / (yHI+yHII+yH2 + yHe + yel)
    return pr/dn * mu * mH_RAMSES / kB_RAMSES


# TODO see if it works in emission.py
# Luminosity field
# Cloudy Intensity obtained assuming height = 1cm
# Return intensity values erg/s/cm**2
# Multiply intensity at each pixel by volume of pixel -> luminosity
def get_luminosity(line):
   def _luminosity(field, data):
      return data['gas', 'flux_' + line]*data['gas', 'volume']
   return copy.deepcopy(_luminosity)


#number density of hydrogen atoms
def _my_H_nuclei_density(field, data):
    dn=data["ramses","Density"].in_cgs()
    XH_RAMSES=0.76 #defined by RAMSES in cooling_module.f90
    YHE_RAMSES=0.24 #defined by RAMSES in cooling_module.f90
    mH_RAMSES=yt.YTArray(1.6600000e-24,"g") #defined by RAMSES in cooling_module.f90

    return dn*XH_RAMSES/mH_RAMSES


def _pressure(field, data):
    if 'hydro_thermal_pressure' in dir(ds.fields.ramses): # and 
        #'Pressure' not in dir(ds.fields.ramses):
        return data['ramses', 'hydro_thermal_pressure']


def _xHI(field, data):
    if 'hydro_xHI' in dir(ds.fields.ramses): # and \
        #'xHI' not in dir(ds.fields.ramses):
        return data['ramses', 'hydro_xHI']


def _xHII(field, data):
    if 'hydro_xHII' in dir(ds.fields.ramses): # and \
        #'xHII' not in dir(ds.fields.ramses):
        return data['ramses', 'hydro_xHII']


def _xHeII(field, data):
    if 'hydro_xHeII' in dir(ds.fields.ramses): # and \
        #'xHeII' not in dir(ds.fields.ramses):
        return data['ramses', 'hydro_xHeII']


def _xHeIII(field, data):
    if 'hydro_xHeIII' in dir(ds.fields.ramses): # and \
        #'xHeIII' not in dir(ds.fields.ramses):
        return data['ramses', 'hydro_xHeIII']

'''
-------------------------------------------------------------------------------
Load Simulation Data
Add Derived Fields
-------------------------------------------------------------------------------
'''

ds = yt.load(filename, extra_particle_fields=epf)

ds.add_field(
    ("gas","number_density"),
    function=_my_H_nuclei_density,
    sampling_type="cell",
    units="1/cm**3",
    force_override=True
)


ds.add_field(
    ("ramses","Pressure"),
    function=_pressure,
    sampling_type="cell",
    units="1",
    #force_override=True
)

ds.add_field(
    ("ramses","xHI"),
    function=_xHI,
    sampling_type="cell",
    units="1",
    #force_override=True
)

ds.add_field(
    ("ramses","xHII"),
    function=_xHII,
    sampling_type="cell",
    units="1",
    #force_override=True
)

ds.add_field(
    ("ramses","xHeII"),
    function=_xHeII,
    sampling_type="cell",
    units="1",
    #force_override=True
)

ds.add_field(
    ("ramses","xHeIII"),
    function=_xHeIII,
    sampling_type="cell",
    units="1",
    #force_override=True
)

ds.add_field(
    ("gas","my_temperature"),
    function=_my_temperature,
    sampling_type="cell",
    # TODO units
    #units="K",
    #units="K*cm**3/erg",
    units='K*cm*dyn/erg',
    force_override=True
)

# Ionization parameter
ds.add_field(
    ('gas', 'ion_param'),
    function=_ion_param,
    sampling_type="cell",
    units="cm**3",
    force_override=True
)

ds.add_field(
    ("gas","my_H_nuclei_density"),
    function=_my_H_nuclei_density,
    sampling_type="cell",
    units="1/cm**3",
    force_override=True
)

# Normalize by Density Squared Flag
dens_normalized = True
if dens_normalized: 
    units = '1/cm**6'
else:
    units = '1'

# Instance of EmissionLineInterpolator for line list at filename
# print(line_list) #see cell 2 above for details
emission_interpolator = EmissionLineInterpolator(lines, line_list) #why is this interpolated? computational speedup?

# Add flux and luminosity fields for all lines in the list
for i, line in enumerate(lines):
    ds.add_field(
        ('gas', 'flux_' + line),
        function=emission_interpolator.get_line_emission(
            i, dens_normalized=dens_normalized
        ),
        sampling_type='cell',
        units=units,
        force_override=True
    )
    # TODO change get_line_emission to accept line not idx

    ds.add_field(
        ('gas', 'luminosity_' + line),
        function=emission_interpolator.get_luminosity(lines[i]),
        #function=get_luminosity(lines[i]),
        sampling_type='cell',
        units='1/cm**3',
        force_override=True
    )
print("Status, fully loaded!")

if __name__ == "__main__":
    # --------------------------------------------
    # INPUT: list of emission lines (exact names)
    # --------------------------------------------
    lines = np.array([
        'H1 6562.80A', 'O1 1304.86A', 'O1 6300.30A', 'O2 3728.80A',
        'O2 3726.10A', 'O3 1660.81A', 'O3 1666.15A', 'O3 4363.21A',
        'O3 4958.91A', 'O3 5006.84A', 'He2 1640.41A', 'C2 1335.66A',
        'C3 1906.68A', 'C3 1908.73A', 'C4 1549.00A', 'Mg2 2795.53A',
        'Mg2 2802.71A', 'Ne3 3868.76A', 'Ne3 3967.47A', 'N5 1238.82A',
        'N5 1242.80A', 'N4 1486.50A', 'N3 1749.67A', 'S2 6716.44A',
        'S2 6730.82A'
    ])
    # --------------------------------------------
    # Convert "O1 1304.86A" → "O1_1304.86A"
    # --------------------------------------------
    lines = np.array([l.replace(" ", "_") for l in lines])

    # -------------------------------------------------------------------------
    # PREP: load all gas data and cell volume once (faster than inside loop)
    # -------------------------------------------------------------------------
    ad = ds.all_data()
    # ----------------------------------------------------
    # OUTPUT ARRAYS
    # ----------------------------------------------------
    total_fluxes = []
    total_luminosities = []

    # creating proj of a boxed region
    cx = np.mean(ad["star", "particle_position_x"])
    cy = np.mean(ad["star", "particle_position_y"])
    cz = np.mean(ad["star", "particle_position_z"])
    center = [cx, cy, cz]
    halfa = ds.quan(20, "kpc")

    low_edge = [center[0] - halfa, center[1] - halfa, center[2] - halfa]
    high_edge = [center[0] + halfa, center[1] + halfa, center[2] + halfa]

    cube_region = ds.region(center, low_edge, high_edge)

    ad_box = cube_region


    cell_volume_box = ad_box["cell_volume"]   # in cm^3
    # ----------------------------------------------------
    # OUTPUT ARRAYS
    # ----------------------------------------------------
    total_fluxes_box = []
    total_luminosities_box = []


    cell_vol  = cell_volume_box.to("cm**3").d[None, :]     # broadcast to shape (1, Ncells)

    # trim this out and use simple array after initial computation
    # lum_fields  = [("gas", f"luminosity_{line}") for line in lines]
    # lum_data  = np.vstack([ad_box[f] for f in lum_fields])  # shape: (Nlines, Ncells) #.to("erg/s").d
    # total_luminosities_box  = lum_data.sum(axis=1).to_value()
    total_luminosities_box = np.array([3.90955858e+41, 3.31949205e+36, 2.26410702e+34, 2.72101305e+36,
    1.20475976e+37, 2.26032755e+37, 8.64213222e+35, 2.53087718e+36,
    9.26514983e+35, 8.30460881e+36, 1.03474205e+42, 4.13492301e+33,
    4.83716059e+35, 1.49603706e+37, 1.26528935e+37, 4.24149818e+29,
    2.19186638e+37, 1.10545778e+37, 5.37404641e+35, 1.63110992e+35,
    6.97985519e+32, 3.48213989e+32, 1.58560762e+33, 1.32286373e+36,
    2.66617959e+36])
    print("Status: Luminosity field loaded")
    zsource = ds.current_redshift
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    DL = cosmo.luminosity_distance(zsource).to("cm") #get luminosity distance
    Fline = total_luminosities_box/4/np.pi/(DL)**2/(1+zsource)
    print("Status: distance correction complete")
    # ============================================================
    # Build Gaussian spectrum in F_lambda (REST FRAME)
    # ============================================================

    def gaussian_spectrum_lambda(lambda_grid_A, line_centers_A, sigma_v_kms, F_lines):
        """
        Returns F_lambda spectrum (erg/s/cm^2/Å)
        """

        lambda_grid_A = np.asarray(lambda_grid_A)
        line_centers_A = np.asarray(line_centers_A)

        # Convert velocity width → wavelength width
        sigma_lambda = line_centers_A * (sigma_v_kms / 3e5)  # Å

        # Gaussian normalization
        A = F_lines / (sigma_lambda * np.sqrt(2*np.pi))  # erg/s/cm^2/Å

        # Broadcast
        dlam = lambda_grid_A[None, :] - line_centers_A[:, None]

        S = A[:, None] * np.exp(-0.5 * (dlam / sigma_lambda[:, None])**2)

        return np.sum(S, axis=0)  # F_lambda


    # High-resolution rest-frame grid
    lambda_rest = np.linspace(200, 10000, 50000)  # Å

    Flambda_rest = gaussian_spectrum_lambda(
        lambda_rest,
        wavelengths,
        sigma_v_kms=50,
        F_lines=Fline
    )


    # ============================================================
    # REDSHIFT (In F_lambda)
    # ============================================================

    lambda_obs = lambda_rest * (1 + zsource)

    Flambda_obs = Flambda_rest / (1 + zsource)


    # ============================================================
    # STEP 3: Convert F_lambda → F_nu
    # ============================================================

    c_cgs = 2.99792458e10  # cm/s

    lambda_obs_cm = lambda_obs * 1e-8

    Fnu_obs = (lambda_obs_cm**2 / c_cgs) * Flambda_obs  # erg/s/cm^2/Hz

    # convert to μJy
    Fnu_obs_uJy = Fnu_obs * 1e23 * 1e6


    # ============================================================
    # STEP 4: Compare with stellar SED, this uses original script by Dr. Ricotti
    # ============================================================

    from sed_continuum import sed_cont

    wav_obs0, wav_obs1, a1, b1, y_tot0, y_tot1, *_ = sed_cont()

    wav_obs0 = np.array(wav_obs0[a1:b1])  # micron
    stellar = y_tot0  # μJy


    # interpolate line spectrum onto stellar grid
    from scipy.interpolate import interp1d

    interp_func = interp1d(
        lambda_obs * 1e-4,   # Å → micron
        Fnu_obs_uJy,
        bounds_error=False,
        fill_value=0
    )

    lines_interp = interp_func(wav_obs0)


    # ============================================================
    # STEP 5: PLOT
    # ============================================================

    plt.figure(figsize=(6,4), dpi=150)

    plt.plot(wav_obs0, stellar, label="Stellar SED")
    plt.plot(wav_obs0, lines_interp, label="Emission lines")

    plt.yscale("log")
    plt.xlim(0.5, 7)
    plt.ylim(1e-5, 1e-1) #toggle this on and off to see the problem

    plt.xlabel(r"$\lambda_{\rm obs}$ [$\mu$m]")
    plt.ylabel(r"$F_\nu$ [$\mu$Jy]")
    plt.legend()

    plt.show()