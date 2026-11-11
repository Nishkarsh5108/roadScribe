"""Source-channel-rate factorisation of vibration spectra (the measuring instrument of the identifiability study).

Not claimed as new: penalised Whittle estimation is Pawitan & O'Sullivan (1994),
and the shifted-template structure is the one used for stellar spectra (Bedell et
al. 2019) and drive-by monitoring at several speeds.
"""
from .whittle import SCRData, SCRFit, bspline_basis, fit_scr, multitaper_frames

__all__ = ["SCRData", "SCRFit", "bspline_basis", "fit_scr", "multitaper_frames"]
