import io
import warnings
import requests
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.coordinates as coord

from astropy.io import fits
from astroquery.sdss import SDSS
from textwrap import dedent

"""
A suite of functions for querying the SDSS
science archive for SDSS-V targeting catalogs
and publicly-available spectra from DR18 and
prior data releases.

Authors: Zach Vanderbosch (Caltech)
Updated: 2023 June 22
"""


# base URLs per data release to retrieve spectra
MAX_DR = 18
DRS_AVAILABLE = [17,18]
BASE_SAS_URLs = {
      'dr17-boss': "https://data.sdss.org/sas/dr17/eboss/spectro/redux/v5_13_2/spectra/full",
    'dr17-legacy': "https://data.sdss.org/sas/dr17/sdss/spectro/redux/26/spectra",
    'dr17-segue2': "https://data.sdss.org/sas/dr17/sdss/spectro/redux/104/spectra",
           'dr18': "https://dr18.sdss.org/sas/dr18/spectro/boss/redux/v6_0_4/spectra/full"
}

# Saved Table containing SDSS-PS1 cross match results
DEFAULT_TABLE = "sdss_archival_xmatch.parquet"


# Supress some annoying np.genfromtext warnings
# when doing SDSS SQL queries
warnings.filterwarnings(
    "ignore", category=np.VisibleDeprecationWarning) 


def _sdss_SPECquery_payload (region_payload, data_release):
    """
    Function to generate full SQL query for SDSS spectra.

    Parameters:
    -----------
    region_payload: str
        String from _sdss_region_payload that
        defines the RA-Dec region being searched
        in the query.
    data_release: int
        SDSS data release value passed from the
        SPECquery_from_sample function.

    Returns:
    --------
    query_payload: str
        The full SQL query.
    """

    if data_release == 17:
        query_payload = dedent("""\
            SELECT
            main.specobjid,
            main.ra,
            main.dec,
            main.programname,
            main.survey,
            main.sourcetype as objtype,
            main.class,
            main.subclass,
            main.plate,
            main.mjd,
            main.fiberid
            FROM SpecObj main"""
        )
    elif data_release == 18:
        query_payload = dedent("""\
            SELECT
            main.specobjid,
            main.plug_ra as ra,
            main.plug_dec as dec,
            main.programname,
            main.survey,
            main.objtype,
            main.class,
            main.subclass,
            main.plate,
            main.mjd,
            main.fiberid,
            main.healpix_dir
            FROM spAll main"""
        )

    query_payload += region_payload

    return query_payload



def _sdss_MOSquery_payload (region_payload, join_tic=False, 
    join_ps1=False, carton=None):
    """
    Function to generate full SQL query for SDSS-V
    MOS targetin catalogs.

    Parameters:
    -----------
    region_payload: str
        String from _sdss_region_payload that
        defines the RA-Dec region being searched
        in the query.
    join_tic: bool
        Whether to join the mos_tic_v8 catalog in
        the SQL query.
    join_ps1: bool
        Whether to jointhe mos_panstarrs1 catalog
        in the SQL query.
    carton: str, int, or None
        Define the carton name (str) or carton ID
        number (int) to only return entries from 
        a specific SDSS-V carton. If None, entries
        from all cartons are returned.

    Returns:
    --------
    query_payload: str
        The full SQL query.
    """

    # The base SQL query payload
    query_payload = dedent("""\
        SELECT
        main.catalogid as catalogid,
        main.ra as ra, 
        main.dec as dec,
        mcar.carton as carton,
        mcar.carton_pk as cartonID,
        mcat.lead as parent_catalog,
        cad.label as cadence,
        inst.label as instrument"""
    )

    join_payload = dedent("""
        FROM mos_carton mcar
        JOIN mos_carton_to_target mctt 
          ON mcar.carton_pk = mctt.carton_pk
        JOIN mos_cadence cad
          ON cad.pk = mctt.cadence_pk
        JOIN mos_instrument inst
          ON inst.pk = mctt.instrument_pk
        JOIN mos_target main 
          ON main.target_pk = mctt.target_pk
        JOIN mos_catalog mcat
          ON mcat.catalogid = main.catalogid"""
    )

    # Join tables with the tic_V8 table
    if join_tic:

        query_payload += ","+dedent("""
            tic.gaia as gaia_sourceid, 
            tic.gaiamag as gaia_g,
            tic.plx as parallax,
            tic.gaiabp as gaia_bp,
            tic.gaiarp as gaia_rp"""
        )

        join_payload += dedent("""
            JOIN mos_catalog_to_tic_v8 c2tic 
              ON c2tic.catalogid = main.catalogid
            JOIN mos_tic_v8 tic 
              ON tic.id = c2tic.target_id"""
        )

    if join_ps1:

        query_payload += ","+dedent("""
            ps1.extid_hi_lo as ps1_id"""
        )

        join_payload += dedent("""
            JOIN mos_catalog_to_panstarrs1 c2ps1 
              ON c2ps1.catalogid = main.catalogid
            JOIN mos_panstarrs1 ps1 
              ON ps1.catid_objid = c2ps1.target_id"""
        )

    if carton is not None:
        if isinstance(carton,str):
            carton_payload = dedent("""
                AND mcar.carton = '{}'""".format(carton))
        elif isinstance(carton,int):
            carton_payload = dedent("""
                AND mcar.carton_pk = {}""".format(carton))
    else:
        carton_payload = ""

    query_payload += join_payload + region_payload + carton_payload

    return query_payload


def _sdss_region_payload(sample_ra, sample_dec, qtype):
    """
    Convenience function to set up the WHERE constraints
    of the SQL query, defining the RA-Dec extents that 
    provide full coverage of the sample.

    Parameters
    ----------
    sample_ra: array
        Array of right ascension coordinates of sample,
        in units of decimal degrees.
    sample_dec: array
        Array of declination coordinates of sample,
        in units of decimal degrees.
    qtype: str
        Must be either 'MOS' or 'SPEC'. Needed to
        determine what column names to use for the
        RA and dec values in each table.

    Returns:
    --------
    region_payload: str
        SQL constraints that define the RA an DEC
        limits of the SQL query.
    """

    # Define search coordinates
    raMin = min(sample_ra)
    raMax = max(sample_ra)
    decMin = min(sample_dec)
    decMax = max(sample_dec)
    buff = 0.1 # 0.1-arcminute buffer to add to query region
    buff = coord.Angle(buff*u.arcmin).to('degree').value

    if qtype == 'MOS':
        rcol = 'ra'
        dcol = 'dec'
    elif qtype == 'SPEC17':
        rcol = 'ra'
        dcol = 'dec'
    elif qtype == 'SPEC18':
        rcol = 'plug_ra'
        dcol = 'plug_dec'
    else:
        raise ValueError(
            f"qtype = {qtype} is not allowed." + \
            "Must be either 'MOS', 'SPEC17', or 'SPEC18'."
        )


    # Formulate rectangular region query
    if raMax - raMin > 300: # crosses 0-360 border
        
        raMin_low = 0.
        raMax_low = max(sample_ra[sample_ra<180])
        raMin_upp = min(sample_ra[sample_ra>180])
        raMax_upp = 360.
        
        region_payload = dedent("""
            WHERE ((main.{:s} between {:.6f} and {:.6f}) 
              AND (main.{:s} between {:.6f} and {:.6f}))
               OR ((main.{:s} between {:.6f} and {:.6f}) 
              AND (main.{:s} between {:.6f} and {:.6f}))""".format(
                rcol, raMin_low, raMax_low+buff, 
                dcol, decMin-buff, decMax+buff,
                rcol, raMin_upp-buff, raMax_upp, 
                dcol, decMin-buff, decMax+buff
            )
        )   
    else: # doesn't cross 0-to-360 degree border
        region_payload = dedent("""
            WHERE ((main.{:s} between {:.6f} and {:.6f}) 
              AND (main.{:s} between {:.6f} and {:.6f}))""".format(
                rcol, raMin-buff, raMax+buff, 
                dcol, decMin-buff, decMax+buff
            )
        )

    return region_payload



def MOSquery_from_sample (sample_ra, sample_dec, join_tic=False, 
    join_ps1=False,carton=None, crossmatch=False, verbose=False):
    """
    Function that performs a query of SDSS-V MOS targets
    that covers the RA-Dec region spanned by the sample 
    of target coordinates.

    Parameters
    ----------
    sample_ra: array
        Array of right ascension coordinates of sample,
        in units of decimal degrees.
    sample_dec: array
        Array of declination coordinates of sample,
        in units of decimal degrees.
    join_tic: bool
        Whether to join the mos_tic_v8 catalog in
        the SQL query.
    join_ps1: bool
        Whether to jointhe mos_panstarrs1 catalog
        in the SQL query.
    carton: str or int
        The SDSS-V carton name or carton ID number.
        If specified, only entries in this carton
        will be returned.
    crossmatch: bool
        Whether to perform a crossmatch between the 
        sample coordinates and the returned SDSS-V
        MOS targets.
    verbose: bool
        If True, prints out the SQL query.

    Returns:
    --------
    res: DataFrame
        Returned entries from the SQL query. None if
        the query was unsuccessful.
    match_idx: array
        Array of indices into the sample coordinates
        that provide the closest match per entry in
        the "res" DataFrame. None if crossmatch=False.
    match_sep: array
        Array of on-sky separations, in arcseconds,
        between each entry in the "res" DataFrame and 
        its closest match. None if crossmatch=False.
    """

    # Generate SQL coordinate constraints
    region_payload = _sdss_region_payload(
        sample_ra, sample_dec, 'MOS')

    # Generate full query payload
    full_query_payload = _sdss_MOSquery_payload(
        region_payload,
        join_tic=join_tic,
        join_ps1=join_ps1,
        carton=carton
    )
    if verbose:
        print(f"\n\n{full_query_payload}\n")


    # Send the query
    try:
        res = SDSS.query_sql(
            full_query_payload, 
            data_release=18
        ).to_pandas()
    except Exception as e:
        print(f"/nException: {e}")
        return None, None, None

    # Decode string columns
    for cname in res.columns:
        if res[cname].dtype == object:
            res[cname] = res[cname].str.decode('utf-8')

    if crossmatch:

        res_coords = coord.SkyCoord(
            ra=res.ra.values*u.deg,
            dec=res.dec.values*u.deg,
            frame='icrs'
        )
        sample_coords = coord.SkyCoord(
            ra=sample_ra*u.deg,
            dec=sample_dec*u.deg,
            frame='icrs'
        )

        idx,dsep,_ = res_coords.match_to_catalog_sky(sample_coords)
        dsep = dsep.arcsec

        return res, idx, dsep

    else: return res, None, None



def SPECquery_from_sample (sample_ra, sample_dec, data_release='all', 
    crossmatch=False, verbose=False):
    """
    Function that performs a query of SDSS sources for
    which spectra have been obtained within the RA-Dec
    region spanned by the sample of target coordinates.

    Parameters
    ----------
    sample_ra: array
        Array of right ascension coordinates of sample,
        in units of decimal degrees.
    sample_dec: array
        Array of declination coordinates of sample,
        in units of decimal degrees.
    data_release: str or int
        Which SDSS data release to query. Defaults to
        'all' which means it will search both DR17 and
        DR18. Otherwise, specificy a specific data
        release, but can only be 17 or 18.
    crossmatch: bool
        Whether to perform a crossmatch between the 
        sample coordinates and the returned SDSS-V
        MOS targets.
    verbose: bool
        If True, prints out the SQL query.

    Returns:
    --------
    res: DataFrame
        Returned entries from the SQL query. None if
        the query was unsuccessful.
    match_idx: array
        Array of indices into the sample coordinates
        that provide the closest match per entry in
        the "res" DataFrame. None if crossmatch=False.
    match_sep: array
        Array of on-sky separations, in arcseconds,
        between each entry in the "res" DataFrame and 
        its closest match. None if crossmatch=False.
    """

    if (isinstance(data_release,int)) & (data_release not in DRS_AVAILABLE):
        print(f'data_release={data_release} is not an allowed value.')
        return None, None, None

    # Determine what data releases to query
    if isinstance(data_release,int):
        drs_to_query = [data_release]
    elif data_release == 'all':
        drs_to_query = DRS_AVAILABLE
    else:
        print(f'data_release={data_release} is not an allowed value.')
        return None, None, None


    qresults = []
    for dr in drs_to_query:

        # Generate SQL coordinate constraints
        region_payload = _sdss_region_payload(
        sample_ra, sample_dec, f'SPEC{dr}')

        # Generate full query payload
        full_query_payload = _sdss_SPECquery_payload(
            region_payload,
            data_release=dr
        )

        if verbose:
            print(f"\n\nDR-{dr}:")
            print(f"\n{full_query_payload}\n")

        # Send the query
        try:
            res = SDSS.query_sql(
                full_query_payload, 
                data_release=dr
            ).to_pandas()
        except Exception as e:
            continue

        # Decode string columns
        for cname in res.columns:
            if res[cname].dtype == object:
                res[cname] = res[cname].str.decode('utf-8')

        # Generate paths to reduced spectra on the SDSS Science Archive

        SAS_spec_urls = []

        if dr == 17:

            for i,row in res.iterrows():

                if row.survey in ['sdss']:
                    base_url = BASE_SAS_URLs['dr17-legacy']
                    fname = f"spec-{row.plate:04d}-{row.mjd}-{row.fiberid:04d}.fits"
                    spec_url = f"{base_url}/{row.plate:04d}/{fname}"

                elif row.survey in ['boss','eboss']:
                    base_url = BASE_SAS_URLs['dr17-boss']
                    fname = f"spec-{row.plate}-{row.mjd}-{row.fiberid:04d}.fits"
                    spec_url = f"{base_url}/{row.plate}/{fname}"

                elif row.survey in ['segue1']:
                    base_url = BASE_SAS_URLs['dr17-legacy']
                    fname = f"spec-{row.plate:04d}-{row.mjd}-{row.fiberid:04d}.fits"
                    spec_url = f"{base_url}/{row.plate:04d}/{fname}"

                elif row.survey in ['segue2']:
                    base_url = BASE_SAS_URLs['dr17-segue2']
                    fname = f"spec-{row.plate:04d}-{row.mjd}-{row.fiberid:04d}.fits"
                    spec_url = f"{base_url}/{row.plate:04d}/{fname}"

                else:
                    print(res.programname.unique())
                    print(res.survey.unique())
                    raise Exception('Error: Program name or Survey not identified.')

                SAS_spec_urls.append(spec_url)

        elif dr == 18:

            base_url = BASE_SAS_URLs['dr18']

            for i,row in res.iterrows():
                fname = row.healpix_dir.split("/")[-1]
                spec_url = f"{base_url}/{row.plate}p/{row.mjd}/{fname}"
                SAS_spec_urls.append(spec_url)
            res.drop('healpix_dir',axis=1,inplace=True)
        
        res['spec_dr'] = [dr]*len(res)
        res['spec_url'] = SAS_spec_urls
        qresults.append(res)


    # Check whether any entries were returned
    if len(qresults) == 0:
        return None, None, None
    else:
        res = pd.concat(qresults,ignore_index=True)

    # Perform crossmatch if desired
    if crossmatch:

        res_coords = coord.SkyCoord(
            ra=res.ra.values*u.deg,
            dec=res.dec.values*u.deg,
            frame='icrs'
        )
        sample_coords = coord.SkyCoord(
            ra=sample_ra*u.deg,
            dec=sample_dec*u.deg,
            frame='icrs'
        )

        idx,dsep,_ = res_coords.match_to_catalog_sky(sample_coords)
        dsep = dsep.arcsec

        return res, idx, dsep

    else: return res, None, None


def get_SDSS_spectrum(spec_url):
    """
    Function to fetch spectrum given the URL to
    dowload it from.

    Parameters:
    -----------
    spec_url: str
        URL pointing to location of spectrum file

    Returns:
    --------
    spec_data: dict or None
        Dictionary containing wavelength, flux,
        flux uncertainty, and sky flux arrays, 
        along with the primary FITS header of 
        the file. Returns None if request fails.
    """

    spec_response = requests.get(spec_url)
    scode = spec_response.status_code
    sreason = spec_response.reason

    if scode != 200:
        print(f'Response status Code {scode}: {sreason}')
        return None

    # Read response content
    with fits.open(io.BytesIO(spec_response.content)) as hdul:
        sdata = hdul[1].data
        shdr = hdul[1].header
        phdr = hdul[0].header

    Nd = len(sdata)
    wave = np.zeros(Nd)
    flux = np.zeros(Nd)
    ivar = np.zeros(Nd)
    fsky = np.zeros(Nd)
    for s in range(Nd):
        wave[s] = 10.0**(sdata[s][1])
        flux[s] = sdata[s][0]
        ivar[s] = sdata[s][2]
        fsky[s] = sdata[s][6]
    eflux = np.sqrt(1./ivar)

    spec_data = {
        'wavelength': wave,
        'flux': flux,
        'e_flux': eflux,
        'sky': fsky,
        'header': phdr
    }

    return spec_data


def match_PS1_sources_to_SDSS(ps1_IDs, sep_tol=3.0, 
    table_name=DEFAULT_TABLE):
    """
    Function to return entries from the SDSS X-match
    table that match provided Pan-STARRS1 (PS1) 
    source IDS. Uses the pyarrow package for speed
    improvements.

    Parameters:
    -----------
    ps1_IDs: int or array-like of ints
        PS1 ID, or list/array of IDs, to check
        for entries within the table.
    table_name: str
        Filename and path to table
    sep_tol: float
        Separation, in arcsec, to consider a
        match close enough.

    Returns:
    --------
    df_matched: DataFrame or None
        Rows from the SDSS X-match table that
        match the provided PS1 IDs within the 
        given sep_tol distance. Returns None 
        if no entries are found.

    """

    import pyarrow.parquet as pq

    if isinstance(ps1_IDs,int):
        ps1_IDs = [ps1_IDs]

    # Use pyarrow to load matching rows
    pydf = pq.read_table(
        table_name, 
        filters=[
            ('nearest_psid','in',ps1_IDs),
            ('ps1_sep','<',sep_tol)
        ]
    )
    df_matched = pydf.to_pandas()

    if len(df_matched.index) > 0:
        df_matched.sort_values(
            by=['nearest_psid','mjd'],
            inplace=True,
            ignore_index=True
        )
        return df_matched
    else:
        return None

