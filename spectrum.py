
class Spectrum:
    def __init__(self, wavelength_um, flux_ujy, frame="observed"):
        self.wavelength_um = wavelength_um
        self.flux_ujy = flux_ujy
        self.frame = frame

    def resample(self, new_grid_um):
        # one interpolation implementation, used everywhere instead of ad-hoc interp1d calls
        ...

    def __add__(self, other):
        # resample onto the finer of the two grids, then add fluxes
        ...

    def plot(self, ax=None, **kwargs):
        ...