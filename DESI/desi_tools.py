import io
import time
import requests
import numpy as np
import pandas as pd

from astropy.io import fits

"""
Tools for downloading DESI 
Early Data Release (EDR) spectra.

Author: Zach Vanderbosch (Caltech)
Updated: 2023 June 28
"""


EDR_MOUNTAIN = "fuji"
DESI_EDR_ROOT = "https://data.desi.lbl.gov/public/edr"
VIZTOOL_EDR_ROOT = "https://www.legacysurvey.org/viewer/desi-spectrum/edr"
SPECTRO_EDR_ROOT = f"{DESI_EDR_ROOT}/spectro/redux/{EDR_MOUNTAIN}/healpix"


def get_DESI_spectrum (targetID, survey, program, healpix,
    download_spectrum=True, coadd=True, show_viz_url=False):
    """
    Function to download and return the coadded or individual 
    spectrum/spectra  from the healpix directory for a single 
    object, given its DESI target-ID, the survey and program
    it observed in, and the healpix number that covers it.

    WARNING: Downloading the spectrum for a single object
    from the DESI data archive is fairly slow, since you 
    must first download a fairly large file (>100 Mb, and
    sometimes >1 Gb) that contains the spectra 
    for all objects within a single DESI HEALpix, to get 
    just the few rows of data corresponding to your 
    object. USING THIS FUNCTION IS NOT RECOMMENDED for 
    downloading spectra for large numbers of objects.


    Parameters:
    -----------
    targetID: int
        The DESI target-ID for your object
    survey: str
        The DESI survey covering your target, 
        e.g. sv1, sv2, sv3, other, etc.
    program: str
        The DESI program that observed your
        target, e.g. dark, bright, backup, etc.
    healpix: int
        The DESI healpix number (nside=64) that
        covers our target.
    download_spectrum: bool
        Whether to download the spectrum.
    coadd: bool
        Whether to download just the coadded 
        spectrum. If False, will instead download
        the "sppectra" file which is considerably
        larger but contains the individual spectra
        that went into the coadd.
    show_viz_url: bool
        Whether to print out the URL that will
        show a interactive plot of your object's
        spectrum.

    Returns:
    --------
    specdata: dict or None
        Dictionary containing the DESI spectrum, or
        spectra, with the following strucure:

        {
            b_wave: array
            r_wave: array
            z_wave: array
            spec_1: dict
            ...,
            spec_N: dict
        }

        where the "wave" entries provide the 
        wavelength arrays for each arm of the
        spectrograph, and the "spec_i" entries are 
        also dictionaries containing the MJD, 
        EXPTIME, flux, and flux uncertainties for
        each spectrum 1 through N correspnding to
        the provided target ID, with the 
        following structure:

        {
            mjd: float      # Only for single spec
            numexp: int     # Only for coadded spec
            exptime: float
            b_flux: array
            b_eflux: array
            r_flux: array
            r_eflux: array
            z_flux: array
            z_eflux: array
        }

        Returns None if download_spectrum=False, 
        or if the query was unsuccessful.
    """

    # Generate the URLs
    pixnum = int(healpix/100)
    healpix_url = f"{SPECTRO_EDR_ROOT}/{survey}/{program}/{pixnum}/{healpix}"
    viz_url = f"{VIZTOOL_EDR_ROOT}/targetid{targetID}"
    if coadd:
        spec_file = f"coadd-{survey}-{program}-{healpix}.fits"
    else:
        spec_file = f"spectra-{survey}-{program}-{healpix}.fits"
    full_url = f"{healpix_url}/{spec_file}"

    # Print out spectrum viewer URL if desired
    if show_viz_url:
        print(f"\nDESI Spectrum Viewer: {viz_url}")

    # Check if download is desired
    if not download_spectrum:
        return None

    # Send request
    print('\nRetrieving DESI spectrum...',end='')
    t0 = time.time()
    r = requests.get(full_url)
    t1 = time.time()
    print('Finished')

    # Print out reponse status
    scode = r.status_code
    sreason = r.reason
    print(f'Response time: {t1-t0:.1f} seconds')
    if scode == 200:
        rsize = float(r.headers['Content-Length'])/(1024**2)
        print(f'Response status code: {scode} SUCCESS')
        print(f'Response size: {rsize:.0f} Mb')
    else:
        print(f'Response status code: {scode} {sreason}')
        return None

    # Read response content
    with fits.open(io.BytesIO(r.content)) as hdul:
        fibermap = hdul['FIBERMAP'].data
        wavedata_b = hdul['B_WAVELENGTH'].data
        fluxdata_b = hdul['B_FLUX'].data
        ivardata_b = hdul['B_IVAR'].data
        wavedata_r = hdul['R_WAVELENGTH'].data
        fluxdata_r = hdul['R_FLUX'].data
        ivardata_r = hdul['R_IVAR'].data
        wavedata_z = hdul['Z_WAVELENGTH'].data
        fluxdata_z = hdul['Z_FLUX'].data
        ivardata_z = hdul['Z_IVAR'].data

    # Get target data
    file_targetIDs = fibermap.TARGETID
    target_idx = np.where(file_targetIDs == targetID)[0]
    Nidx = len(target_idx)

    if Nidx == 0:
        return None

    # Initialize specdata dictionary with wavelength
    # arrays. Wavelengths are always the same for each
    # spectrograph arm.
    specdata = {
        'b_wave':wavedata_b,
        'r_wave':wavedata_r,
        'z_wave':wavedata_z
    }
    for i,tidx in enumerate(target_idx):

        if coadd:
            numexp = fibermap.COADD_NUMEXP[tidx]
            exptime = fibermap.COADD_EXPTIME[tidx]
        else:
            mjd = fibermap.MJD[tidx]
            exptime = fibermap.EXPTIME[tidx]
        
        bflux = fluxdata_b[tidx,:]
        bivar = ivardata_b[tidx,:]
        rflux = fluxdata_r[tidx,:]
        rivar = ivardata_r[tidx,:]
        zflux = fluxdata_z[tidx,:]
        zivar = ivardata_z[tidx,:]

        key = f"spec_{i+1:d}"
        if coadd:
            data = {
                'numexp':numexp,
                'exptime':exptime,
                'b_flux':bflux,
                'b_eflux':np.sqrt(1./bivar),
                'r_flux':rflux,
                'r_eflux':np.sqrt(1./rivar),
                'z_flux':zflux,
                'z_eflux':np.sqrt(1./zivar)
            }
        else:
            data = {
                'mjd':mjd,
                'exptime':exptime,
                'b_flux':bflux,
                'b_eflux':np.sqrt(1./bivar),
                'r_flux':rflux,
                'r_eflux':np.sqrt(1./rivar),
                'z_flux':zflux,
                'z_eflux':np.sqrt(1./zivar)
            }
        specdata[key] = data

    return specdata

