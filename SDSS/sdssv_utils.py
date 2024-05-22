import os
import tqdm
import time
import warnings
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import sdss_access as sdss

from sys import exit
from netrc import netrc
from functools import partial
from multiprocessing import Pool
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.visualization import PercentileInterval
from urllib.request import HTTPPasswordMgrWithDefaultRealm
from urllib.request import HTTPBasicAuthHandler
from urllib.request import build_opener, install_opener, urlopen
from urllib.error import HTTPError

warnings.filterwarnings("ignore",category=RuntimeWarning)
warnings.filterwarnings("ignore",category=UserWarning)

"""
A suite of functions used for downloading 
and visualizing SDSS-V spectra.

Zach Vanderbosch (Caltech)
Last updated 2024 January 10
"""

SAS_BASE = "https://data.sdss5.org/sas"
MAX_THREADS = 16 # Maximum number of parallel download threads for SAS queries

# Remote spAll Filenames on SAS by version, type, and coadd
SAS_SPALL_NAMES = {
    'master': {
        'daily': {
            'lite': "spAll-lite-master.fits.gz",
            'full': "spAll-master.fits.gz"}},
    'v6_0_9': {
        'daily': {
            'lite': "spAll-lite-v6_0_9.fits.gz",
            'full': "spAll-v6_0_9.fits"}},
    'v6_1_0': {
        'daily': {
            'lite': "spAll-lite-v6_1_0.fits.gz",
            'full': "spAll-v6_1_0.fits"}},
    'v6_1_1': {
        'daily': {
            'lite': "spAll-lite-v6_1_1.fits.gz",
            'full': "spAll-v6_1_1.fits.gz"},
        'epoch': {
            'lite': "spAll-lite-v6_1_1.fits.gz",
            'full': "spAll-v6_1_1.fits.gz"},
        'allepoch': {
            'lite': "spAll-lite-v6_1_1-allepoch.fits.gz",
            'full': "spAll-v6_1_1-allepoch.fits.gz"}},
    'v6_1_2': {
        'daily': {
            'lite': "spAll-lite-v6_1_2.fits.gz", 
            'full': "spAll-v6_1_2.fits.gz"},
        'epoch': {
            'lite': "spAll-lite-v6_1_2.fits.gz",
            'full': "spAll-v6_1_2.fits.gz"},
        'allepoch': {
            'lite': "spAll-lite-v6_1_2-allepoch.fits.gz",
            'full': "spAll-v6_1_2-allepoch.fits.gz"}},
    'v6_1_3': {
        'daily': {
            'lite': "spAll-lite-v6_1_3.fits.gz", 
            'full': "spAll-v6_1_3.fits.gz"},
        'epoch': {
            'lite': "spAll-lite-v6_1_3.fits.gz",
            'full': "spAll-v6_1_3.fits.gz"},
        'allepoch': {
            'lite': "spAll-lite-v6_1_3-allepoch.fits.gz",
            'full': "spAll-v6_1_3-allepoch.fits.gz"}},
    'ipl-1': {
        'daily': {
            'lite': "spAll-lite-v6_0_9.fits.gz",
            'full': "spAll-v6_0_9.fits"}},
    'ipl-2': {
        'daily': {
            'lite': "spAll-lite-v6_0_9.fits.gz",
            'full': "spAll-v6_0_9.fits"}},
    'ipl-3': {
        'daily': {
            'lite': "spAll-lite-v6_1_3.fits.gz", 
            'full': "spAll-v6_1_3.fits.gz"},
        'epoch': {
            'lite': "spAll-lite-v6_1_3.fits.gz",
            'full': "spAll-v6_1_3.fits.gz"},
        'allepoch': {
            'lite': "spAll-lite-v6_1_3-allepoch.fits.gz",
            'full': "spAll-v6_1_3-allepoch.fits.gz"}}
}

# Expected local spAll filenames after downloading and decompressing
LOCAL_SPALL_NAMES = {
    'master': {
        'daily': {
            'lite': "spAll-lite-master.fits",
            'full': "spAll-master.fits"}},
    'v6_0_9': {
        'daily': {
            'lite': "spAll-lite-v6_0_9.fits",
            'full': "spAll-v6_0_9.fits"}},
    'v6_1_0': {
        'daily': {
            'lite': "spAll-lite-v6_1_0.fits",
            'full': "spAll-v6_1_0.fits"}},
    'v6_1_1': {
        'daily': {
            'lite': "spAll-lite-v6_1_1.fits",
            'full': "spAll-v6_1_1.fits"},
        'epoch': {
            'lite': "spAll-lite-v6_1_1.fits",
            'full': "spAll-v6_1_1.fits"},
        'allepoch': {
            'lite': "spAll-lite-v6_1_1-allepoch.fits",
            'full': "spAll-v6_1_1-allepoch.fits"}},
    'v6_1_2': {
        'daily': {
            'lite': "spAll-lite-v6_1_2.fits", 
            'full': "spAll-v6_1_2.fits"},
        'epoch': {
            'lite': "spAll-lite-v6_1_2.fits",
            'full': "spAll-v6_1_2.fits"},
        'allepoch': {
            'lite': "spAll-lite-v6_1_2-allepoch.fits",
            'full': "spAll-v6_1_2-allepoch.fits"}},
    'v6_1_3': {
        'daily': {
            'lite': "spAll-lite-v6_1_3.fits", 
            'full': "spAll-v6_1_3.fits"},
        'epoch': {
            'lite': "spAll-lite-v6_1_3.fits",
            'full': "spAll-v6_1_3.fits"},
        'allepoch': {
            'lite': "spAll-lite-v6_1_3-allepoch.fits",
            'full': "spAll-v6_1_3-allepoch.fits"}},
    'ipl-1': {
        'daily': {
            'lite': "spAll-lite-v6_0_9.fits",
            'full': "spAll-v6_0_9.fits"}},
    'ipl-2': {
        'daily': {
            'lite': "spAll-lite-v6_0_9.fits",
            'full': "spAll-v6_0_9.fits"}},
    'ipl-3': {
        'daily': {
            'lite': "spAll-lite-v6_1_3.fits",
            'full': "spAll-v6_1_3.fits"},
        'epoch': {
            'lite': "spAll-lite-v6_1_3.fits",
            'full': "spAll-v6_1_3.fits"},
        'allepoch': {
            'lite': "spAll-lite-v6_1_3-allepoch.fits",
            'full': "spAll-v6_1_3-allepoch.fits"}
    }
}

# Pipeline versions used by each IPL
IPL_BOSS_PIPELINE_VERSION = {
    'ipl-1':'v6_0_9',
    'ipl-2':'v6_0_9',
    'ipl-3':'v6_1_3'
}
IPL_APOGEE_PIPELINE_VERSION = {
    'ipl-1':'1.0',
    'ipl-2':'1.0',
    'ipl-3':'1.3'
}
IPL_ASTRA_PIPELINE_VERSION = {
    'ipl-1':'0.2.6',
    'ipl-2':'0.3.0',
    'ipl-3':'0.5.0'
}


def get_IPL_BOSS_pipeline_version(ipl_version):
    """
    Function to retrieve BOSS pipeline version used
    for a given Internal Product Launch (IPL)
    """
    pipe_version = IPL_BOSS_PIPELINE_VERSION[ipl_version]
    return pipe_version

def get_IPL_ASTRA_pipeline_version(ipl_version):
    """
    Function to retrieve Astra pipeline version used
    for a given Internal Product Launch (IPL)
    """
    pipe_version = IPL_ASTRA_PIPELINE_VERSION[ipl_version]
    return pipe_version

def get_IPL_APOGEE_pipeline_version(ipl_version):
    """
    Function to retrieve APOGEE pipeline version used
    for a given Internal Product Launch (IPL)
    """
    pipe_version = IPL_APOGEE_PIPELINE_VERSION[ipl_version]
    return pipe_version

def get_spall_filename(version, spall_type, coadd_type):
    """
    Function to retrieve expected local spAll filename
    given the version and spAll type.
    """
    spall_filename = LOCAL_SPALL_NAMES[version][coadd_type][spall_type]
    return spall_filename

def get_access(netfile,remote_base):
    """
    Function to establish remote connection authentication
    """

    # Load in SDSS-V username/password from the .netrc file
    netkey = "data.sdss5.org"
    netrc_content = netrc(netfile)
    authenticators = netrc_content.authenticators(netkey)
    username = authenticators[0]
    password = authenticators[2]

    # Connect and Log in to the SAS
    passman = HTTPPasswordMgrWithDefaultRealm()
    passman.add_password(None, remote_base, username, password)
    authhandler = HTTPBasicAuthHandler(passman)
    opener = build_opener(authhandler)
    install_opener(opener)
    


def load_file(url, savefile, netfile, rbase, pbar=False, overwrite=False):
    """
    Function to perform the urllib request and
    save returned content to local file.

    Parameters:
    -----------
    url: str
        URL from which to download spectrum
    savefile: str
        Path + filename to save downloaded data to
    netfile: str
        Path + filename for .netrc file
    rbase: str
        Base remote directory
    pbar: bool
        Whether to print out a download progress bar
    overwrite: bool
        Whether to overwrite existing file

    Returns:
    --------
    filesize: int
        Saved file size in bytes
    """

    # Check for existing file. Skip download if file exists and overwrite=False
    if not overwrite:
        if os.path.isfile(savefile):
            filesize = os.stat(savefile).st_size
            return filesize
    
    # Establish access
    get_access(netfile,rbase)

    specfile = url.split("/")[-1]
    try:
        r = urlopen(url)
        meta = r.info()
        meta_func = meta.get_all
        meta_length = meta_func("Content-Length")
        file_size = int(meta_length[0])
    except HTTPError as e:
        r = None
        filesize = 0
        print("HTTP error code %r." % e.code)
        return

    block_sz = 8192
    if r is not None:
        if pbar:
            print(f'Downloading file of size {file_size/1024**2:.0f} Mb...')
            with tqdm.tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024, desc='Progress') as pbar:
                with open(savefile, 'wb') as handle:
                    while True:
                        buffer = r.read(block_sz)
                        if not buffer:
                            break
                        pbar.update(len(buffer))
                        handle.write(buffer)
                filesize = os.stat(savefile).st_size
        else:
            with open(savefile, 'wb') as handle:
                while True:
                    buffer = r.read(block_sz)
                    if not buffer:
                        break
                    handle.write(buffer)
            filesize = os.stat(savefile).st_size
    
    return filesize


def download_spec_sdssaccess(spall_subset, version, spectype='full',
    release='sdsswork'):
    """
    Function for downloading proprietary SDSS-V spectra
    from the Science Archive Server using the sdss-access
    package.

    Parameters:
    -----------
    spall_subset: DataFrame
        Subset of the spAll file for which you want
        to download spectra.
    version: str
        The pipeline version to use, e.g. 'v6_1_0'
    spectype: str
        Either 'full' (coadd + individual exposures)
        or 'lite' (coadd only)
    release: str
        Which SDSS data release to download from. 
        'sdsswork' is the release for proprietary
        SDSS-V data, but you can also download public
        data using 'DR18', 'DR17', etc.'

    Returns:
    --------
    spec_paths: list
        List of local paths to downloaded spectra.

    """

    access = sdss.Access(release=release)
    access.remote()

    # Set sdss-access spectype variable
    if spectype == 'full':
        sdss_access_spectype = 'specFull'
    elif spectype == 'lite':
        sdss_access_spectype = 'specLite'

    # Add each matched spectrum
    for i,row in spall_subset.iterrows():
        access.add(
            sdss_access_spectype, 
            catalogid=row.CATALOGID, 
            fieldid=row.FIELD, 
            mjd=row.MJD, 
            run2d=version)

    # Set up the stream and perform download
    access.set_stream()
    access.commit()

    # Get local paths to downloaded spectra
    spec_paths = access.get_paths()

    return spec_paths



def download_spec_urllib(spall_subset, version, save_dir='.', coadd_type='daily',
    spectype='full', netfile=None, threads=4, chunksize=100, sleeptime=1.0,
    overwrite=False):
    """
    Function for downloading proprietary SDSS-V spectra
    from the Science Archive Server using the urllib package
    and multiprocessing. The core urllib code was taken 
    from the sdss-access source code.

    Parameters:
    -----------
    spall_subset: DataFrame
        Subset of the spAll file for which you want
        to download spectra.
    version: str
        The pipeline version to use, e.g. 'v6_1_3', 'ipl-3', 'master'
    save_dir: str
        Directory to save downloaded spectra into.
    coadd_type: str
        One of the following: 
        - 'daily' (daily coadds)
        - 'epoch' (multi-day coadds, only available with certain IPLs/pipelines)
        - 'allepoch' (all-epoch coadds, only available with certain IPLs/pipelines)
    spectype: str
        One of the following: 
        - 'full' (daily coadd + individual exposures)
        - 'lite' (daily coadd only)
    netfile: str
        Path to the .netrc file containing the SDSS-V
        username and password. If None, will search for
        the .netrc file in your home directory. See
        https://sdss-access.readthedocs.io/en/latest/auth.html
        for instructions on how to create the .netrc file.
    threads: int
        Number of parallel download streams to use.
    chunksize: int
        Number of downloads to hold in the 
        multiprocessing pool at once. Allows for more useful
        progress bar updates when downloading a large
        number of spectra.
    sleeptime: float
        Time in seconds to pause between each chunk of jobs
        sent to the SDSS Science Archive Server.
    overwrite: bool
        Whether to overwrite existing spectra with the same
        filename as one about to be downloaded.

    Returns:
    --------
    savefiles: list
        List of local paths to downloaded spectra.
    """

    Nspec = len(spall_subset)
    if Nspec == 0:
        return []
    else:
        print(f'{Nspec} new spectrum/spectra to Download')

    # Make sure threads is not too large
    if threads > MAX_THREADS:
        threads = MAX_THREADS

    # Set remote directory URL with desired version
    if 'ipl' in version:
        pipe_version = get_IPL_BOSS_pipeline_version(version)
        remote_base = f"{SAS_BASE}/{version}"
        remote_dir = f"{remote_base}/spectro/boss/redux/{pipe_version}"
    else:
        remote_base = f"{SAS_BASE}/sdsswork"
        remote_dir = f"{remote_base}/bhm/boss/spectro/redux/{version}"

    # Generate spec URLs and file names
    urls = []
    savefiles = []
    for i,row in spall_subset.iterrows():

        # Get SDSS-V identifying info
        field = row.FIELD
        mjd = row.MJD
        specfile = row.SPEC_FILE.strip()

        # Generate url and filename            
        if coadd_type == 'epoch':
            spec_url = f"{remote_dir}/epoch/spectra/{spectype}/{field:06d}/{mjd:5d}/{specfile}"
        elif coadd_type == 'allepoch':
            spec_url = f"{remote_dir}/spectra/{spectype}/allepoch/{mjd:5d}/{specfile}"
        else:
            spec_url = f"{remote_dir}/spectra/{spectype}/{field:06d}/{mjd:5d}/{specfile}"
        save_name = f"{save_dir}/{specfile}"
        urls.append(spec_url)
        savefiles.append(save_name)


    # Check for .netrc file
    if netfile is None:
        netfile = f"{os.path.expanduser('~')}/.netrc"
    if not os.path.isfile(netfile):
        exit(f'{netfile} DOES NOT EXIST!')

    # Perform downloads. Does the download in chunks
    # in order to have a useful progress bar.
    num_chunks = int(np.ceil(Nspec/chunksize))
    bar_fmt = "    {l_bar}{bar:40}{r_bar}{bar:-40b}"
    with tqdm.tqdm(total=Nspec, bar_format=bar_fmt) as pbar:
        for i in range(num_chunks):

            # Get indices defining the chunk
            idx_low = int(i*chunksize)
            idx_upp = int((i+1)*chunksize)
            if idx_upp > Nspec:
                idx_upp = Nspec

            # Get chunk URLs and savefiles
            urls_chunk = urls[idx_low:idx_upp]
            savefiles_chunk = savefiles[idx_low:idx_upp]

            # Perform the download
            pool_args = [
                (x,y,netfile,remote_base,False,overwrite) 
                for x,y in zip(urls_chunk, savefiles_chunk)
            ]
            with Pool(threads) as p:
                for _ in p.starmap(partial(load_file),pool_args):
                    pass
            
            # Update progress bar
            pbar.update(len(urls_chunk))

            # Pause submissions
            if i < num_chunks-1:
                time.sleep(sleeptime)

    return savefiles



def download_spall(version, spall_type='lite', coadd_type='daily', save_dir='.', 
    netfile=None, overwrite=False):
    """
    Function for downloading proprietary SDSS-V spectra
    from the Science Archive Server using the urllib package
    and multiprocessing. The core urllib code was taken 
    from the sdss-access source code.

    Parameters:
    -----------
    version: str
        The pipeline version to use, e.g. 'v6_1_0' or 'ipl-3'
    spall_type: str
        The type, lite or full, of spAll file to download.
    coadd_type: str
        Which coadd files to download: daily, epoch, or allepoch
    save_dir: str
        Directory to save downloaded spAll file into.
    spectype: str
        Either 'full' (coadd + individual exposures)
        or 'lite' (coadd only)
    netfile: str
        Path to the .netrc file containing the SDSS-V
        username and password. If None, will search for
        the .netrc file in your home directory. See
        https://sdss-access.readthedocs.io/en/latest/auth.html
        for instructions on how to create the .netrc file.
    overwrite: bool
        Whether to overwrite an existing spall file
    """

    # Set remote directory URL with desired version
    if 'ipl' in version:
        pipe_version = get_IPL_BOSS_pipeline_version(version)
        remote_base = f"{SAS_BASE}/{version}"
        remote_dir = f"{remote_base}/spectro/boss/redux/{pipe_version}"
        if coadd_type == 'epoch':
            remote_dir += "/epoch"
    else:
        remote_base = f"{SAS_BASE}/sdsswork"
        remote_dir = f"{remote_base}/bhm/boss/spectro/redux/{version}"

    # Check for .netrc file
    if netfile is None:
        netfile = f"{os.path.expanduser('~')}/.netrc"
    if not os.path.isfile(netfile):
        exit(f'{netfile} DOES NOT EXIST!')

    # Establish remote connection
    get_access(netfile,remote_base)

    # Define URL and filename
    sas_name = SAS_SPALL_NAMES[version][coadd_type][spall_type]
    local_name = LOCAL_SPALL_NAMES[version][coadd_type][spall_type]
    spall_url = f"{remote_dir}/{sas_name}"
    spall_filename = f"{save_dir}/{sas_name}"
    spall_decompressed_filename = f"{save_dir}/{local_name}"

    # Check for existing Decompressed SPALL file
    if os.path.isfile(spall_decompressed_filename) and not overwrite:
        print(f"Local spAll file found at {spall_decompressed_filename}")
        print(f"Set overwrite=TRUE to re-download this file.")
        return

    # Download file
    load_file(
        spall_url, 
        spall_filename, 
        netfile, 
        remote_base, 
        pbar=True, 
        overwrite=overwrite
    );

    # Uncompress the spAll lite file
    if os.path.isfile(spall_filename):
        if '.gz' in spall_filename:
            print('Decompressing spAll file...')
            time.sleep(1)
            subprocess.run(['gunzip','-f',spall_filename])

    print('\nspAll download finished.\n')

    return




def get_scatter(wave, flux):
    """
    Function to get quick estimate of spectral
    scatter within a wavelength window.
    """

    wave_range = wave[(wave>5600) & (wave<5800)]
    flux_range = flux[(wave>5600) & (wave<5800)]

    median_flux = np.nanmedian(flux_range)
    flux_diff = abs(flux_range - median_flux)
    mad_flux = np.nanmedian(flux_diff)

    return mad_flux



def plot_spectrum(spec_data, spall_dat=None, ax=None, show_sky=False,
    show_uncertainty=False):
    """
    Function for generating quick plots of SDSS-V spectra

    Parameters:
    -----------
    spec_data: dict
        A dictionary generated by the load_SDSS_spectrum
        function with spectral data.
    spall_dat: DataFrame
        The spAll row for this object.
    ax: Axes object
        The plotting axis to use.
    show_sky: bool
        Whether to plot the sky flux.
    show_uncertainty:
        Whether to plot the flux uncertainty.

    Returns:
    --------
    ax: Axes object
        The plotting axis used.

    """

    if ax is None:
        fig = plt.figure(figsize=(14,5))
        ax = fig.add_subplot(111)

    # Get some metadata
    texp_total = spec_data['texp']
    nexp_total = spec_data['nexp']
    mjd_exp = spec_data['mjd']
    shdr = spec_data['header']
    try:
        extname = shdr['EXTNAME']
    except:
        extname = 'COADD'
    
    # Plot spectrum, trimming the edges a little bit
    ax.plot(
        spec_data['wavelength'][100:], 
        spec_data['flux'][100:], 
        c='C7', lw=1, label=f'Object ({extname})'
    )
    if show_sky:
        ax.plot(
            spec_data['wavelength'][100:], 
            spec_data['sky'][100:], 
            c='coral', lw=1, alpha=0.75, zorder=-1, 
            label='Sky'
        )
    if show_uncertainty:
        ax.plot(
            spec_data['wavelength'][100:], 
            spec_data['e_flux'][100:], 
            c='skyblue', lw=1, alpha=0.75, zorder=-1, 
            label='Uncertainty'
        )
    ax.legend(fontsize=13,handletextpad=0.2,loc='upper right')

    # Add title
    if spall_dat is not None:
        specname = spall_dat.SPEC_FILE
        ra = spall_dat.FIBER_RA
        dec = spall_dat.FIBER_DEC
        try:
            gaia_mag = spall_dat.GAIA_G
        except:
            try:
                gaia_mag = spall_dat.GAIA_G_MAG
            except Exception:
                traceback.print_exc()
        sname_short = specname.split("/")[-1]
        title = f"{sname_short}\nG = {gaia_mag:.2f} mag, RA = {ra:.5f}, Dec = {dec:.5f}" + \
                f", Texp = {texp_total:.0f}s, Nexp = {nexp_total:d}, MJD = {mjd_exp:.6f}"
    else:
        title = f"Texp = {texp_total:.0f}s, Nexp = {nexp_total:d}, MJD = {mjd_exp:.6f}"
    ax.set_title(title,fontsize=15);


    # Add axis labels
    ax.set_xlabel('Wavelength ($\mathrm{\AA}$)',fontsize=14)
    ax.set_ylabel('$f_{\lambda}$   ($10^{-17}$ $\mathrm{erg/s/cm^2/\AA}$)',fontsize=14)

    # Set XY limits
    PI = PercentileInterval(99.)
    flux_scatter = get_scatter(spec_data['wavelength'],spec_data['flux'])
    flux_range = PI.get_limits(spec_data['flux'][100:-100])
    y_lowlim = flux_range[0] - 13.0*flux_scatter
    y_upplim = flux_range[1] + 26.0*flux_scatter
    ax.set_ylim(y_lowlim, y_upplim)

    # Set grid and tick params
    ax.grid(ls=":", c='silver')
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.tick_params(which='both',top=True,right=True,direction='in',labelsize=13)
    plt.show()

    return ax




def load_SDSS_spectrum(filename, method='COADD'):
    """
    Function to load spectrum given the local filename.

    Parameters:
    -----------
    filename: str
        Path + filename for spectrum.
    method: str
        Allowed Values: COADD, MULTI, or ALL
        Determines which spectra to extract and return 
        from the FITS file.

    Returns:
    --------
    spec_data: dict or None
        Dictionary containing wavelength, flux,
        flux uncertainty, and sky flux arrays, 
        along with the primary FITS header of 
        the file. Returns None if file does not
        exist.
    """

    if not os.path.isfile(filename):
        return None

    # Load FITS file
    specdata = {}
    headers = {}
    with fits.open(filename) as hdul:
        num_ext = len(hdul)
        for i in range(num_ext):
            
            if i == 0: # Skip primary HDU
                phdr = hdul[i].header
                continue
                
            hdr = hdul[i].header
            extname = hdr['EXTNAME']
            if 'COADD' in extname: # Coadded spectrum
                dat = hdul[i].data
                headers[extname] = hdr
                specdata[extname] = dat
            elif 'MJD_EXP' in extname: # Individual spectra
                dat = hdul[i].data
                headers[extname] = hdr
                specdata[extname] = dat
            elif 'SPALL' in extname:
                spall_dat = hdul[i].data
                num_exp = spall_dat.NEXP[0]
                texp_total = spall_dat.EXPTIME[0]
            else:
                continue

    if method == 'COADD':

        sdata = specdata['COADD']
        Nd = len(sdata)
        loglam = sdata.LOGLAM
        wave = 10**(loglam)
        flux = sdata.FLUX
        ivar = sdata.IVAR
        fsky = sdata.SKY
        eflux = np.sqrt(1./ivar)
        try:
            mjd_list = [float(x)/86400 for x in spall_dat.TAI_LIST[0].split(" ")]
        except:
            mjd_list = [float(x)/86400 for x in spall_dat.TAI_LIST[0][0].split(" ")]
                
        mjd_mean = np.mean(mjd_list)

        spec_data = {
            'COADD':{
                'nexp': num_exp,
                'texp': texp_total,
                'mjd': mjd_mean,
                'wavelength': wave,
                'flux': flux,
                'e_flux': eflux,
                'sky': fsky,
                'header': phdr,
            }
        }
    elif method == 'ALL':

        spec_data = {}
        for ext in headers.keys():

            sdata = specdata[ext]
            Nd = len(sdata)
            loglam = sdata.LOGLAM
            wave = 10**(loglam)
            flux = sdata.FLUX
            ivar = sdata.IVAR
            fsky = sdata.SKY
            eflux = np.sqrt(1./ivar)
            if 'COADD' in ext:
                hdr_exp = phdr
                texp_exp = texp_total
                nexp_exp = num_exp
                try:
                    mjd_list = [float(x)/86400 for x in spall_dat.TAI_LIST[0].split(" ")]
                except:
                    mjd_list = [float(x)/86400 for x in spall_dat.TAI_LIST[0][0].split(" ")]
                mjd_exp = np.mean(mjd_list)
            else:
                hdr_exp = headers[ext]
                texp_exp = hdr_exp['EXPTIME']
                nexp_exp = 1
                mjd_exp = hdr_exp['TAI-BEG']/86400

            spec_data[ext.strip()] = {
                'nexp': nexp_exp,
                'texp': texp_exp,
                'mjd': mjd_exp,
                'wavelength': wave,
                'flux': flux,
                'e_flux': eflux,
                'sky': fsky,
                'header': hdr_exp
            }
    else:
        print(f"Method = {method} not supported.")
        return None

    return spec_data




def singleObject_cross_match(ra, dec, catalog_coord, radius=3.0):
    """
    Function to perform a cross match between a single source
    position and a catalog (e.g. the SDSS-V spAll file).

    Parameters:
    -----------
    ra: float
        Search RA in decimal degrees
    dec: float
        Search Dec in decimal degrees
    catalog_coord: SkyCoord object
        Coordinates of catalog sources to search within
    radius: float
        Search radius in arcsec

    Returns:
    --------
    idx_match: array
        Boolean array matching the length of catalog_coord
        that is True where a match within the search radius
        is found.
    """

    # Create coordinate object for source
    object_coord = SkyCoord(
        ra=ra*u.deg,
        dec=dec*u.deg,
        frame='icrs'
    )

    # Find catalog matches on sky
    idx_match = catalog_coord.separation(object_coord) < radius*u.arcsec

    return idx_match


def multiObject_cross_match(ra, dec, catalog_coord, radius=3.0):
    """
    Function to perform a cross match between multiple source
    positions and a catalog (e.g. the SDSS-V spAll file).

    Parameters:
    -----------
    ra: array
        Array of object RA in decimal degrees
    dec: array
        Array of object Dec in decimal degrees
    catalog_coord: SkyCoord object
        Coordinates of catalog sources to search within
    radius: float
        Search radius in arcsec

    Returns:
    --------
    idx_sources: array
        Indices into ra and dec for matched sources
    idx_catalog: array
        Indicis into catalog_coord for matched sources
    """

    # Create coordinate object for source
    object_coords = SkyCoord(
        ra=ra*u.deg,
        dec=dec*u.deg,
        frame='icrs'
    )

    # Find catalog matches on sky
    idx_sources,idx_catalog,_,_ = catalog_coord.search_around_sky(object_coords, radius*u.arcsec)

    return idx_sources, idx_catalog





def sky_plot(coords, size=None, ax=None, galplot=True, color='coral', label=None):
    """
    Function to generate a sky position plot with the 
    provided coordinates.

    Parameters:
    -----------
    coords: SkyCoord object
        Object coordinates to plot
    size: int
        Number of coordinates to plot
    ax: matplotlib Axes object
        Axes object to use for plotting
    galplot: bool
        Whether to overplot the galactic plane
        and center positions.
    color: str
        Matplotlib plotting color
    label: str
        Legend label

    Returns:
    --------
    ax: matplotlib Axes object
        The plotting axis used
    """


    # Display Sky Positions of random selection of SDSS-V Spectra
    if ax is None:
        fig = plt.figure(figsize=(14,8))
        ax = fig.add_subplot(111,projection='aitoff')

    # Source coordinates
    ra_rad = coords.ra.wrap_at(180*u.deg).rad
    dec_rad = coords.dec.rad
    Ncoord = len(coords)

    # Make sure size parameter makes sense
    if size is not None:
        if size > Ncoord:
            size = Ncoord

    # Galactic plane coordinates
    gal_l = np.linspace(0.0,360,1000)
    gal_b = np.zeros(len(gal_l))
    gc = SkyCoord(l=gal_l*u.deg,b=gal_b*u.deg,frame='galactic')
    gal_ra = gc.icrs.ra.wrap_at(180 * u.deg).radian
    gal_dec = gc.icrs.dec.radian
    gal_dec = gal_dec[np.argsort(gal_ra)]
    gal_ra = gal_ra[np.argsort(gal_ra)]

    # Galactic Center coordinates
    sag = SkyCoord("17h45m40.04s −29d00m28.1s",frame='icrs')
    sag_ra = sag.ra.wrap_at(180 * u.deg).radian
    sag_dec = sag.dec.radian

    if size is None: # Plot all sources
        ax.scatter(-ra_rad, dec_rad, s=2, fc=color,ec=color,label=label)
    else:
        idx_arr = np.arange(Ncoord,dtype=int)
        sample_idx = np.random.choice(idx_arr, size=size, replace=False)
        ax.scatter(-ra_rad[sample_idx], dec_rad[sample_idx], s=2, fc=color,ec=color,label=label)

    if galplot:
        ax.plot(-gal_ra,gal_dec,c='k')
        ax.scatter(-sag_ra,sag_dec,marker='o',s=80,fc='None', ec='k',lw=2)


    # Add grid and tick labels
    ax.grid(ls=":", c='silver')
    ax.set_axisbelow(False)
    ax.tick_params(which='both',labelsize=12)
    ax.set_xticks(
        [-2.61799388, -2.0943951 , -1.57079633, 
         -1.04719755, -0.52359878,  0.        ,  
          0.52359878,  1.04719755,  1.57079633,  
          2.0943951 ,  2.61799388]
    )
    base_labels = ['210','240','270','300','330','0','30','60','90','120','150']
    xtick_labels = [f'{x}\N{DEGREE SIGN}' for x in base_labels]
    ax.set_xticklabels(xtick_labels[::-1])

    return ax


def sdssv_carton_query(carton_id, qlimit=500000):
    """
    Function to query SDSS Science Archive Server
    for source metadata within a specific SDSS-V carton.

    Parameters:
    -----------
    carton_id: int
        SDSS-V carton ID number
    qlimit: int
        Maximum number of sources to return

    Returns:
    --------
    res: matplotlib Axes object
        The plotting axis used
    """

    from astroquery.sdss import SDSS

    # First get true count of entries matching carton
    catalog_query = """
    SELECT COUNT(*) AS count
    FROM mos_carton mc
    JOIN mos_carton_to_target mctt 
      ON mc.carton_pk = mctt.carton_pk
    JOIN mos_target mt 
      ON mt.target_pk = mctt.target_pk
    WHERE mc.carton_pk = {:d}
    """.format(carton_id)
    
    cres = SDSS.query_sql(catalog_query, data_release=18)
    count = cres[0]['count']
    print("{} Sources in Carton {}".format(count, carton_id))
    
    # Form the SQL query, limit to qlimit sources
    catalog_query = """
    SELECT TOP {:d}
      mc.carton as carton,
      mc.carton_pk as cartonID,
      cad.label as cadence,
      inst.label as instrument,
      mt.catalogid as catalogid, 
      mt.ra as ra, 
      mt.dec as dec, 
      tic.gaia as gaia_sourceid, 
      tic.gaiamag as gaia_g,
      tic.plx as parallax,
      tic.gaiabp as gaia_bp,
      tic.gaiarp as gaia_rp
    FROM mos_carton mc
    JOIN mos_carton_to_target mctt 
      ON mc.carton_pk = mctt.carton_pk
    JOIN mos_cadence cad
      ON cad.pk = mctt.cadence_pk
    JOIN mos_instrument inst
      ON inst.pk = mctt.instrument_pk
    JOIN mos_target mt 
      ON mt.target_pk = mctt.target_pk
    JOIN mos_catalog_to_tic_v8 c2tic 
      ON c2tic.catalogid = mt.catalogid
    JOIN mos_tic_v8 tic 
      ON tic.id = c2tic.target_id
    WHERE mc.carton_pk = {:d}
    ORDER BY NEWID()
    """.format(qlimit,carton_id)

    print('Performing query...',end='',flush=True)
    res = SDSS.query_sql(catalog_query, data_release=18)
    res = res.to_pandas()
    res['carton'] = res['carton'].str.decode('utf-8')
    res['cadence'] = res['cadence'].str.decode('utf-8')
    res['instrument'] = res['instrument'].str.decode('utf-8')
    print('Finished')

    return res


def print_SDSSV_cartons(cartons='all'):
    """
    Function to print out all SDSS-V carton names and IDs

    Parameters:
    -----------
    cartons: str
        What group of cartons to pritn out. 
        Must be 'all', 'mwm', or 'bhm'
    """

    from astroquery.sdss import SDSS

    carton_query = """
    SELECT mc.carton, mc.carton_pk, mm.label, mcat.label
    FROM mos_carton mc
    JOIN mos_mapper mm ON mm.pk = mc.mapper_pk
    JOIN mos_category mcat ON mcat.pk = mc.category_pk
    WHERE mc.mapper_pk > -1
    AND mc.mapper_pk < 2
    """
    
    tmp = SDSS.query_sql(carton_query, data_release=18)
    all_carton_ids = tmp['carton_pk'].value

    if cartons.lower() in ['mwm','all']:
        print('Milky Way Mapper (MWM) Cartons: ')
        print('--------------------------------')
        print('Name                       iD   \n')
        for row in tmp.iterrows():
            if row[2] == 'MWM':
                print("{:26s} {}".format(row[0], row[1]))
    
    if cartons.lower() in ['bhm','all']:
        print('\nBlack Hole Mapper (BHM) Cartons:         ')
        print('-------------------------------------------')
        print('Name                                  iD   \n')
        for row in tmp.iterrows():
            if row[2] == 'BHM':
                print("{:37s} {}".format(row[0], row[1]))
