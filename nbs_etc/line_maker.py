from sim_load import sim_load
import numpy as np
from numpy import ndarray
from astropy import units as u
from astropy import constants as const

def line_generator(wav_ray, lmbds, sigmas, lums): #maybe make ad ds????
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
    
    gaussian_spectrum_from_integrated_flux_erg(wav_ray, lmbds, sigmas, lums)