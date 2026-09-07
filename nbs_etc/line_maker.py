from sim_load import sim_load
import numpy as np
from numpy import ndarray
from astropy import units as u
from astropy import constants as const

def line_generator(wav0, filename=None, ds=None, wavelengths=None): #maybe make ad ds????
    if filename is None and ds is None:
        ds, wavelengths = sim_load()
    elif filename is not None:
        ds, wavelengths = sim_load(filename=filename)
    elif ds is None and wavelengths is not None or ds is None and wavelengths is not None:
        #TODO: Make this a real error
        print("Please pass both ds and wavelengths for full functionality")

    #helped
    # need to make some kind of caching for wavelengths once I get the 
    for wl in wavelengths:
        # Dynamically build regex: matches 'luminosity_', any string, the float formatted to 2 decimals, then 'A'
        # We escape the dot (\.) so regex treats it as a literal decimal point
        wl_str = f"{wl:.2f}".replace(".", r"\.")
        pattern = re.compile(rf"^luminosity_.*{wl_str}A$")
        
        # Search ds.derived_field_list for a matching field tuple
        # field[0] is the field type ('gas'), field[1] is the field name string
        matches = [field for field in ds.derived_field_list if pattern.match(field[1])]
        
        if len(matches) == 1:
            lum_fields.append(matches[0])
        elif len(matches) > 1:
            print(f"Ambiguous matches for wavelength {wl}: {matches}")
        else:
            print(f"No field found matching wavelength {wl}")

    # 2. Extract and sum data exactly like your previous layout
    # (Using ds.all_data() or your specific data container grid/box object)
    ad = ds.all_data()
    lum_data = np.vstack([ad[f] for f in lum_fields])  # shape: (Nlines, Ncells)
    total_luminosities_box = lum_data.sum(axis=1).to_value()
    
    
    # TODO: Add distance correction
    # TODO: Explain unknown process in sigma_v to sigma_nu

    # Function for luminosity diming with redshift

    # Function for dist correction

    # funciton for wavelength shift

    # Determine series of functions that must be executed to create line spectrum
    

    def ergcm2s_to_jyhz(F_erg_cm2_s):
        # 1 erg/s/cm^2 = 1e23 Jy Hz

        return F_erg_cm2_s * 1e23 * u.Jansky

    def gaussian_spectrum_from_integrated_flux_erg(
        lambda_array: ndarray,          # wavelength grid (Å)
        line_cents_array: ndarray,      # line centers (Å)
        sigma_v_kms: ndarray,           # Gaussian sigma in km/s (scalar or array)
        F_line_erg_cm2_s: ndarray       # assuming integrated fluxes are in correct unit: (erg/s/cm^2), array 
    ) -> u.uJy:
        """
        Returns the combined line spectrum S(λ) in Jy.
        Gaussian profile defined in frequency domain but evaluated on λ grid.
        """

        # ---- Convert inputs to arrays ----
        lambda_array = np.asarray(lambda_array, float)        # (Nλ,)
        line_cents_array = np.asarray(line_cents_array, float)  # (Nlines,)
        sigma_v_kms = np.asarray(sigma_v_kms, float)            # scalar or (Nlines,)
        F_line_erg_cm2_s = np.asarray(F_line_erg_cm2_s, float)  # (Nlines,)

        # ---- Convert wavelength to frequency ----
        # Å → cm
        lam_cm = lambda_array * 1e-8
        lam0_cm = line_cents_array * 1e-8

        # λ → ν conversion
        nu_array = const.c.cgs.to_value() / lam_cm         # Hz, shape (Nλ,)
        nu0 = const.c.cgs.to_value()  / lam0_cm             # Hz, shape (Nlines,)

        # ---- Convert integrated flux to (Jy Hz) ----
        F_jyhz = ergcm2s_to_jyhz(F_line_erg_cm2_s)  # shape (Nlines,)

        # ---- Convert sigma_v → sigma_nu ---- # frankly don't know how this works
        sigma_nu = (nu0 / 2.99792458e5) * sigma_v_kms   # km/s to Hz

        # ---- Gaussian amplitude in Jy ----
        A_jy = F_jyhz / (sigma_nu * np.sqrt(2*np.pi))

        # ---- Broadcast Gaussian calculation ----
        # nu_array: (Nλ,) → (1, Nλ)
        # nu0:      (Nlines,) → (Nlines, 1)
        nu_diff = nu_array[None, :] - nu0[:, None]

        S_each = A_jy[:, None] * np.exp(-0.5 * (nu_diff / sigma_nu[:, None])**2)

        # ---- Sum over all lines ----
        S_total = np.sum(S_each, axis=0)

        return S_total.to(u.uJy)
    lum0 = gaussian_spectrum_from_integrated_flux_erg(wav0, wavelengths, 100, total_luminosities_box)
    z = ds.current_redshift()
    flux = lum0/(4*np.pi*d_L**2) * (1+z)
    wav0 = wav0 * (1+z)
    return wav0, flux
if __name__ == "__main__":
    ds, wavelengths = sim_load()
    line_generator(wav0, ds=ds, wavelengths=wavelengths)