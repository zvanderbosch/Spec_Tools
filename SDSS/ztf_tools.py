import io
import requests
import warnings

from astropy import wcs
from astropy.io import fits

# Suppress WCS FITSFixedWarning
warnings.filterwarnings(
    "ignore", category=wcs.FITSFixedWarning)


ZTF_BASEURL = 'https://irsa.ipac.caltech.edu/ibe/data/ztf/products'


def download_ZTF_refimage(fieldID, ccdID, quadID, filtername):
    """
    Function to download a ZTF quadrant reference image

    Parameters:
    -----------
    fieldID: int
        ZTF field ID
    ccdID: int
        ZTF CCD ID (1-16)
    quadID: int
        ZTF quadrant ID (1-4)
    filtername: str
        Filter name (g, r, or i)

    Returns:
    --------
    img: array
        ZTF image data
    hdr:
        ZTF primary image header
    img_wcs: WCS object
        Image WCS object
    """
    
    refim_name = 'ztf_{:06d}_z{}_c{:02d}_q{:1d}_refimg.fits'.format(
        fieldID,filtername,ccdID,quadID)

    ztf_url = '{}/ref/000/field{:06d}/z{}/ccd{:02d}/q{:1d}/{}'.format(
        ZTF_BASEURL,fieldID,filtername,ccdID,quadID,refim_name)

    # Download the ZTF reference image
    print('Downloading ZTF Reference Image...',end='')
    r = requests.get(ztf_url)
    print('Finished')

    # Read response content
    with fits.open(io.BytesIO(r.content)) as hdul:
        img = hdul[0].data
        hdr = hdul[0].header
    img_wcs = wcs.WCS(hdr)

    return img, hdr, img_wcs