import os
import pickle
from pathlib import Path
import pylab as pl

from astropy import units
import astropy.io.fits as fits
import healpy as hp
from joblib import Parallel, delayed
import numpy as np
from numpy.typing import NDArray
import reproject
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

NSIDE = 2048
TAU_IDX = 0
HIQ_IDX = 1
HIU_IDX = 2
GQ_IDX = 3
GU_IDX = 4
DATA_DIR = Path('data/')


# –––––––––––––––––––––––––––––––––––– ADAPTED FROM FORSE CODE –––––––––––––––––––––––––––––––––––––

def set_header(ra, dec, size_pixel, Npix=128, rot=False):
    """ **Adapted from ForSE code**.
    Set the header of a fits file. This is useful for the reprojection.
    We assume to reproject square patches.

    Args:
        ra (float): Longitudinal coordinate of the patch we want to reproject.
        dec (float): Latitudinal coordinate of the patch we want to reproject.
        size_pixel (astropy.quantity[u.deg]): Size of the pixel in degrees.
        Npix (int): Number of pixels on one side of the patch.

    Returns:
        fits.Header: Fits header related to the coordinates of the centroid.

    ZEA stands for "Zenithal Equal Area" projection. It is a
    type of projection that preserves the area of the sky
    regions while minimizing distortion near the center of
    the projection.
    """

    hdr = fits.Header()
    hdr.set("SIMPLE", "T")  # standard FITS
    hdr.set("BITPIX", -32)  # 32-bit floating point
    hdr.set("NAXIS", 2)
    hdr.set("NAXIS1", Npix)
    hdr.set("NAXIS2", Npix)
    hdr.set("CRVAL1", ra)
    hdr.set("CRVAL2", dec)
    hdr.set("CRPIX1", Npix / 2.0 + 0.5)
    hdr.set("CRPIX2", Npix / 2.0 + 0.5)
    if not rot:
        hdr.set("CD1_1", size_pixel)  # pixel scale in degrees
        hdr.set("CD2_2", -size_pixel)
        hdr.set("CD2_1", 0.0000000)  # no rotation
        hdr.set("CD1_2", -0.0000000)
    else:
        # cos(45) = sin(45) = 1/sqrt(2)
        hdr.set("CD1_1", size_pixel * np.cos(np.radians(45)))
        hdr.set("CD1_2", size_pixel * np.cos(np.radians(45)))
        hdr.set("CD2_1", size_pixel * np.cos(np.radians(45)))
        hdr.set("CD2_2", -size_pixel * np.cos(np.radians(45)))
    hdr.set("CTYPE1", "GLON-ZEA")
    hdr.set("CTYPE2", "GLAT-ZEA")
    hdr.set("CUNIT1", "deg")
    hdr.set("CUNIT2", "deg")
    hdr.set("COORDSYS", "galactic")  # international celestial reference system
    return hdr


def h2f(hmap, target_header, coord_in="G"):
    """ **Adapted from ForSE code**.
    Projects a healpix map to flatsky. Interface to the `reproject.reproject_from_healpix`.

    Args:
        hmap (array): Healpix map.
        target_header (fits.Header): The output of `set_header`.
        coord_in (str): Coordinate frame of the input map, 'C' for Celestial, 'E' for Ecliptical,
                        'G' for Galactic.

    Returns:
        array: The reprojected map in a flat pixelization.
    """
    pr, _ = reproject.reproject_from_healpix(
        (hmap, coord_in),
        target_header,
        nested=False,
    )
    return pr


def f2h(flat, target_header, nside, coord_in="G"):
    """ **Adapted from ForSE code**.
    Project flatsky map to healpix. Interface to the `reproject.reproject_to_healpix`.

    Args:
        flat (np.ndarray):
            Map in flat coordinate.
        target_header (astropy.io.fits.Header):
            The output of `set_header`.
        nside (int):
            The output healpix pixelization parameter.
        coord_in (str, optional):
            Coordinate frame for the output map.
            'C' for Celestial, 'E' for ecliptical,
            'G' for Galactic. Defaults to 'G'.

    Returns:
        tuple:
            - pr (np.ndarray): The reprojected map in a healpix pixelization.
            - footprint (np.ndarray): The footprint of the reprojected map as a binary healpix mask.
    """
    pr, footprint = reproject.reproject_to_healpix(
        (flat, target_header),
        coord_system_out=coord_in,
        nside=nside,
        nested=False,
    )
    return pr, footprint


def f2h_fast(flat, target_header, nside, coord_in="G"):
    """ Adaptation of the above method that uses `reproject.healpix.core.patch_to_healpix` instead.
    Much faster as it avoids projecting pixels outside the patch.

    Args:
        flat (np.ndarray):
            Map in flat coordinate.
        target_header (astropy.io.fits.Header):
            The output of `set_header`.
        nside (int):
            The output healpix pixelization parameter.
        coord_in (str, optional):
            Coordinate frame for the output map.
            'C' for Celestial, 'E' for ecliptical,
            'G' for Galactic. Defaults to 'G'.

    Returns:
        tuple:
            - pr (np.ndarray): The reprojected map in a healpix pixelization.
            - footprint (np.ndarray): The footprint of the reprojected map as a binary healpix mask.
    """
    pr, footprint = reproject.reproject_patch_to_healpix(
        (flat, target_header),
        coord_system_out=coord_in,
        nside=nside,
        nested=False,
    )
    return pr, footprint


def get_lonlat_adaptive(size_patch, overlap):
    """ **Adapted from ForSE code**.
    Divides the whole sphere into patches with sizes given by `size_patch`(in degrees) and with
    a certain overlap in degrees. To avoid the fact that the number of patches overlapping increases
    at the poles and decreases at the equator, we implemented an adaptive division of the sphere
    having less overlapping patches  at the poles.

    Args:
        size_patch (astropy.quantity[u.deg]): the angular size of the patches
        overlap (astropy.quantity[u.deg]): the size of the overlap between patches

    Returns:
        tuple[np.array, np.array]: A tuple of lon and lat coordinates
    """
    Nlon = np.int_(np.ceil(360.0 * units.deg / (size_patch - overlap)).value)
    Nlat = np.int_(np.ceil(180.0 * units.deg / (size_patch - overlap)).value)
    if Nlat % 2 == 0:
        Nlat += 1
    offset_lon = 0
    offset_lat = -90
    lat_array = np.zeros(np.int_(Nlat))
    lon_array = np.zeros(np.int_(Nlon))

    lat_array[: Nlat // 2] = [
        offset_lat + ((size_patch).value - overlap.value) * i for i in range(Nlat // 2)
    ]
    lat_array[Nlat // 2 + 1:] = [
        -offset_lat - ((size_patch).value - overlap.value) * i for i in range(Nlat // 2)
    ][::-1]
    lat_array[Nlat // 2] = 0
    lon_array[:Nlon] = [
        offset_lon + ((size_patch).value - overlap.value) * i for i in range(Nlon)
    ]
    Nloneff = np.int_(np.cos(np.radians(lat_array)) * Nlon)
    Nloneff[0] = 5
    Nloneff[-1] = 5
    jumps = np.int_(np.ceil(Nlon / Nloneff) - 1)
    jumps[1] -= 1
    jumps[-2] -= 1
    jumps[Nlat // 2] = 1
    lon, lat = pl.meshgrid(lon_array, lat_array)
    lonj = []
    latj = []
    for kk in range(Nlat):
        lonj.append(lon[kk, :: jumps[kk]])
        latj.append(lat[kk, :: jumps[kk]])
    lonj = np.concatenate(lonj)
    latj = np.concatenate(latj)
    return lonj, latj

# –––––––––––––––––––––––––––––––––– END ADAPTED FROM FORSE CODE –––––––––––––––––––––––––––––––––––


def get_full_sky_maps():
    gi, gq, gu, res_map = hp.read_map(
        DATA_DIR / 'maps/COM_CompMap_IQU-thermaldust-gnilc-varres_2048_R3.00.fits',
        field=(0, 1, 2, -1)
    )
    res_map = np.round(hp.ud_grade(res_map, NSIDE), 0)
    _, hiq, hiu = hp.ud_grade(
        hp.read_map(DATA_DIR / 'maps/IQU_integrated_HI4PI_Hessian_n13top16.fits', field=(0, 1, 2)),
        NSIDE
    )
    hiu *= -1  # flip sign to match Planck convention
    tau = hp.read_map(DATA_DIR / 'maps/HFI_CompMap_ThermalDustModel_2048_R1.20.fits', field=0)
    return gi, gq, gu, tau, hiq, hiu, res_map


def get_patch_size_deg(res: float, multiple=5) -> float:
    return multiple * res / 60


def get_patch_size_rad(res: float, multiple=5) -> float:
    return np.radians(multiple * res / 60)


def change_res(alm, f, t):
    """Smooth alm from f to t arcmin resolution."""
    fwhm = np.sqrt(t**2 - f**2)
    alm_s = hp.smoothalm(alm, fwhm=np.radians(fwhm / 60), inplace=False)
    return hp.alm2map(alm_s, NSIDE)


def generate_centers(res_map, remove_overlaps=True, multiple=20, npix=320):
    """ Generates patch centers for a given full-sky resolution mask. Takes ~11 mins.

    Args:
        res_map (NDArray): A [pixels] array where each pixel is the resolution of the map.
        remove_overlaps (bool): Whether to remove overlapping centers. If True will keep higher
            resolutions. Set to False if generating training patches (more samples), and True if
            generating centers for inference (less samples).
        multiple (int): The multiple of the resolution to use as the patch size, ie
            patchsize = multiple * res (in arcmin).
        npix (int): The number of pixels per side for each patch.

    Returns:
        dict[int, dict[str, NDArray]]: A dictionary mapping resolution to a dictionary containing
            'lon' and 'lat' keys with the corresponding pixel coordinates.
    """
    centers = {}
    pixs = np.arange(hp.nside2npix(NSIDE))
    resols = np.unique(res_map)
    for i, res in tqdm(enumerate(resols), desc='resolutions', position=0):
        # step 1: generate centers for each resolution and clip latitudes/longitudes
        patchsize = get_patch_size_deg(res, multiple)
        # TODO consider taking overlap_pct as an argument -- allows for different overlaps for train
        # and inference
        overlap = 0.6 * patchsize  # i.e., - 2 * lores
        lon, lat = get_lonlat_adaptive(
            patchsize * units.deg, overlap=overlap * units.deg
        )
        # make latitude cuts
        pix_lon, pix_lat = hp.pix2ang(NSIDE, pixs[res_map == res], lonlat=True)
        mask = (
            # remove points above max latitude for given res
            (lat <= pix_lat.max())
            # remove points below min latitude for given res
            & (lat >= pix_lat.min())
            # remove galactic plane
            & (np.abs(lat) > 5)
        )
        lon, lat = lon[mask], lat[mask]
        # longitude cuts post lat mask -- speeds up computation
        pix_mask = (pix_lat <= lat.max()) & (pix_lat >= lat.min())
        mask = (
            # remove points above max longitude for given res
            (lon <= pix_lon[pix_mask].max())
            # remove points below min longitude for given res
            & (lon >= pix_lon[pix_mask].min())
        )
        lon, lat = lon[mask], lat[mask]

        # step 2: remove centers that are too close to each other (keep highest res)
        perpix = patchsize / npix
        buffer = (resols[i + 1] - res) / 2 if i < len(resols) - 1 else 1

        def filter_centers(x, y):
            header = set_header(x, y, perpix, npix)
            # some smoothing happens in the conversion (ie flatproj doesn't have discrete
            # resolutions)
            flatproj = h2f(res_map, header)
            if (
                np.isnan(flatproj).any()  # don't we lose info here?
                # Guarantees that patch centers for a given res will have that res as the lowest res
                # (highest arcmin) in patch.
                or (res + buffer < np.max(flatproj))
                or ((res > np.max(flatproj)) and remove_overlaps)
            ):
                return None
            return x, y

        filtered = Parallel(n_jobs=8)(
            delayed(filter_centers)(x, y)
            for x, y in tqdm(
                zip(lon, lat), desc=f'centers {int(res)}\'', position=1, leave=False, total=len(lon)
            )
        )

        if sum([x is not None for x in filtered]) == 0:
            print(f'No centers for res {res}')
            continue

        filtered = filter(lambda x: x is not None, filtered)
        filtered_lon, filtered_lat = list(zip(*filtered))
        centers[res] = {'lon': filtered_lon, 'lat': filtered_lat}

    print(f'Generated a total of {sum([len(v["lon"]) for v in centers.values()])} centers')
    return centers


def get_smoothed_maps(maps, res_map) -> dict[int, NDArray]:
    """ Generates smoothed full sky maps at different minimum resolutions.

    Args:
        maps (NDArray): A [3, pixels] array where the first dimension is i, q, and u.
        res_map (NDArray): A [pixels] array where each pixel is the resolution of the map.

    Returns:
        dict[int, NDArray]: A dictionary mapping resolution to a [3, pixels] array containing the
            smoothed i, q, and u maps at the given resolution.
    """
    smoothed = {}
    ress = [5, 7, 10, 15, 20, 28, 30, 40, 60, 80, 120, 240]
    sm_res_map = hp.smoothing(res_map, fwhm=np.radians(1))

    prev = maps.copy()

    with open(DATA_DIR / 'smoothed_maps/smoothed_map_5_amin.pkl', 'wb') as f:
        pickle.dump(prev, f)
        smoothed[5] = prev.copy()
    for f, t in tqdm(
        zip(ress[:-1], ress[1:]), desc='Smoothing Maps', position=0, total=len(ress) - 1
    ):
        # generate/load one by one because of memory constraints (map2alm leaks memory like crazy)
        if os.path.exists(DATA_DIR / f'smoothed_maps/smoothed_map_{int(t)}_amin.pkl'):
            with open(DATA_DIR / f'smoothed_maps/smoothed_map_{int(t)}_amin.pkl', 'rb') as f:
                smoothed[t] = pickle.load(f)
            print(f'Loaded smoothed map for {int(t)}\'')
        else:
            map_alm = hp.map2alm(prev)  # supports channel dim
            smoothed[t] = change_res(map_alm, f, t)
            sm_res_map = sm_res_map.clip(f)  # ensures blending math makes sense
            smoothed[t] = (
                np.minimum((sm_res_map - f) / (t - f), 1) * maps
                + np.maximum((t - sm_res_map) / (t - f), 0) * smoothed[t]
            )
            # George's original code -- kept for reference (had no clipping of sm_res_map)
            # for i in range(50331648):
            #     # blend smoothed map where GM < t with the original map where GM > t
            #     smoothed[t][:, i] = (
            #         np.min([(sm_res_map[i] - f) / (t - f), 1]) * maps[:, i]
            #         + np.max([(t - sm_res_map[i]) / (t - f), 0]) * smoothed[t][:, i]
            #     )
            with open(DATA_DIR / f'smoothed_maps/smoothed_map_{int(t)}_amin.pkl', 'wb') as f:
                pickle.dump(smoothed[t], f)

        prev = smoothed[t].copy()

    return smoothed


# DEPRECATED: see implementation in `gen_dataset.py`
def get_patches(tau, hiq, hiu, gi, gq, gu, res_map, centers, multiple=20, npix=320):
    """ Generates 5-channel patches from full sky maps. These patches are used as input to the model
    for generation of a full-sky multiresolution map.

    Args:
        tau (NDArray): Tau full sky map
        hiq (NDArray): HI Q full sky map
        hiu (NDArray): HI U full sky map
        gi (NDArray): GNILC I full sky map
        gq (NDArray): GNILC Q full sky map
        gu (NDArray): GNILC U full sky map
        res_map (NDArray): GNILC full sky resolution mask
        centers (dict[int, dict[str, NDArray]]): Patch centers for each resolution
        multiple (int): The multiple of the resolution to use as the patch size, ie
            patchsize = multiple * res (in arcmin).
        npix (int): The number of pixels per side for each patch

    Returns:
        NDArray: A [N, 5, 80, 80] array where N is the number of patches and the 5 channels
            are tau, q_hi, u_hi, q_lr, u_lr, with the appropriate smoothing applied to tau, q_lr,
            and u_lr.
    """
    smoothed_maps = get_smoothed_maps(np.stack([gi, gq, gu], axis=0), res_map)
    patches = {res: [] for res in centers.keys()}
    for res in tqdm(centers.keys(), desc='resolutions', position=0):
        lon = centers[res]['lon']
        lat = centers[res]['lat']
        pixsize = get_patch_size_deg(res, multiple) / npix
        for x, y in tqdm(
            zip(lon, lat), desc=f'patches {int(res)}\'', position=1, leave=False, total=len(lon)
        ):
            header = set_header(x, y, pixsize, npix)
            patches[res].append(
                np.stack([
                    h2f(tau, header),
                    h2f(hiq, header),
                    h2f(hiu, header),
                    h2f(smoothed_maps[res][1], header),
                    h2f(smoothed_maps[res][2], header)
                ], axis=0)
            )
        patches[res] = np.stack(patches[res], axis=0)
    return patches


def get_apo_masks(resols, apo_pct=0.3, multiple=20, npix=320) -> dict[int, NDArray]:
    masks = {}
    a = np.arange(npix)
    a = np.minimum(a, npix - a - 1)
    # distance to nearest edge per pixel
    pixel_dist = np.minimum(a[:, None], a[None, :])
    for res in resols:
        eta_apo = get_patch_size_rad(res, multiple) * apo_pct
        apo_mask = np.ones((npix, npix))
        pixsize = np.radians(20 * res / 60 / npix)
        x = np.sqrt((1 - np.cos(pixsize * pixel_dist)) / (1 - np.cos(eta_apo)))
        apo_mask[x < 1] = 0.5 * (1 - np.cos(np.pi * x[x < 1]))
        masks[res] = apo_mask
    return masks


def patches_to_full_sky(patches, centers, out_nside=4096, multiple=20, npix=320):
    """ Reprojects flats patches into full sky maps.

    Args:
        patches (dict[int, NDArray]): A dict of res -> [N, 2, 80, 80] arrays where the channel
            dimension is Q and U
        centers (dict[int, dict[str, NDArray]]): A dict of res ->
            {'lon': NDArray, 'lat': NDArray}
        out_nside (int): The nside of the output full sky map
        multiple (int): The multiple of the resolution to use as the patch size, ie
            patchsize = multiple * res (in arcmin)
        npix (int): The number of pixels per side for each patch

    Returns:
        NDArray: A [2, pixels] array representing the full sky where the first dimension is Q
            and U
    """
    out_pix = hp.nside2npix(out_nside)
    new_map = np.zeros((2, out_pix))
    overlap_map = np.zeros(out_pix)
    effective_overlap_map = np.zeros(out_pix)
    reproj_patch = np.empty_like(new_map)
    reverse_res = sorted(patches.keys(), reverse=True)
    apo_masks = get_apo_masks(reverse_res, multiple)
    for res in tqdm(reverse_res, desc='resolutions', position=0):
        patch_arr = patches[res]
        lon = centers[res]['lon']
        lat = centers[res]['lat']
        pixsize = get_patch_size_deg(res, multiple) / npix
        apo_mask = apo_masks[res]
        assert len(lon) == len(lat) == len(patch_arr), "Mismatch in patch centers and patches"
        for i, (x, y) in tqdm(
            enumerate(zip(lon, lat)), desc=f'patches {int(res)}\'', position=1, leave=False,
            total=len(lon)
        ):
            header = set_header(x, y, pixsize, npix)
            reproj_apo = f2h_fast(apo_mask[None, ...], header, out_nside)[0]
            # batch dim
            patch = patch_arr[i]
            # NOTE:
            # f2h is extremely inefficient, especially for large out_nside. Each patch affects only
            # ~1% of pixels and yet huge arrays are allocated every single call. Consider fixing
            # by passing around the footprint and only updating the relevant pixels.
            # A first step would be to remove patches before-hand
            # TODO use f2h_fast --> adapt to remove channel dim
            # reproj_patch, footprint = f2h(patch, header, out_nside)
            for c in range(patch.shape[0]):
                reproj_patch, footprint = f2h_fast(patch[c], header, out_nside)
                # reproj_patch = np.where(
                #     np.isnan(reproj_patch) | np.isinf(reproj_patch), 0, reproj_patch
                # )
                # no nans or infs
                assert not np.isnan(reproj_patch).any() and not np.isinf(reproj_patch).any()
                new_map[c, footprint] = (
                    new_map[c, footprint] * (1 - reproj_apo)
                    + reproj_patch * reproj_apo
                )
            # reproj_patch = np.where(
            #     np.isnan(reproj_patch) | np.isinf(reproj_patch), 0, reproj_patch
            # )
            # no nans or infs
            # assert not np.isnan(reproj_patch).any() and not np.isinf(reproj_patch).any()
            # new_map[:, footprint] = (
            #     new_map[:, footprint] * (1 - reproj_apo)
            #     + reproj_patch * reproj_apo
            # )
            overlap_map[footprint] += 1
            effective_overlap_map[footprint] = (
                effective_overlap_map[footprint] * (1 - reproj_apo)
                + reproj_apo
            )

    return new_map, overlap_map, effective_overlap_map


# a patch of 20 arcmin has angular size 400 arcmin --> model outputs 5 arcmin at 400 arcmin angular
# size but model expects 100 arcmin angular size.
def tile_patch(patch: NDArray) -> NDArray:
    """ Breaks a patch into 49 overlapping patches, each covering 1/4 of the original patch.

    Args:
        patch (NDArray): A [*, 80, 80] patch

    Returns:
        NDArray: A [7, 7, *, 80, 80] array of overlapping patches
    """
    # points = tuple(np.arange(x) for x in patch.shape)
    # upsample = RegularGridInterpolator(points, patch, method='cubic')
    # new_pix_idx = np.linspace(0, 79, 320)
    # upsampled = upsample(
    #     tuple(np.meshgrid(*points[:-2], new_pix_idx, new_pix_idx, indexing='ij'))
    # )
    upsampled = np.repeat(np.repeat(patch, 4, axis=-1), 4, axis=-2)
    upsampled = gaussian_filter(upsampled, sigma=1.81 * 4, order=0, mode='reflect', axes=(-1, -2))
    W = 80
    stride = 40
    tiles = np.empty((7, 7) + patch.shape)
    for i in range(7):
        for j in range(7):
            tiles[i, j] = upsampled[..., i * stride: i * stride + W, j * stride: j * stride + W]
    return tiles


def get_tile_masks() -> NDArray:
    """ Generates 49 masks for the overlapping patches generated by tile_patch.

    Returns:
        NDArray: A [7, 7, 80, 80] array of masks
    """
    x = (
        np.cos(np.concatenate((np.arange(40)[::-1], np.arange(40))) * np.pi / (2 * 39))
        * np.cos(np.concatenate((np.arange(40)[::-1], np.arange(40))) * np.pi / (2 * 39))
    )
    # center mask
    mask_cc = np.zeros((80, 80))
    for i in range(80):
        for j in range(80):
            mask_cc[i, j] = x[i] * x[j]

    # northwest corner mask
    mask_nw = np.copy(mask_cc)
    for i in range(40):
        for j in range(40):
            mask_nw[i, j] = 1
    for i in range(40):
        for j in range(40, 80):
            mask_nw[i, j] = x[j]
    for i in range(40, 80):
        for j in range(40):
            mask_nw[i, j] = x[i]
    mask_sw = np.rot90(mask_nw)  # ccw rotation
    mask_se = np.rot90(mask_sw)
    mask_ne = np.rot90(mask_se)

    # northern edge mask
    mask_nn = np.copy(mask_cc)
    for i in range(40):
        for j in range(80):
            mask_nn[i, j] = x[j]
    mask_ww = np.rot90(mask_nn)
    mask_ss = np.rot90(mask_ww)
    mask_ee = np.rot90(mask_ss)

    return np.stack(
        [np.stack([mask_nw] + [mask_nn]*5 + [mask_ne], axis=0)]
        + [np.stack([mask_ww] + [mask_cc]*5 + [mask_ee], axis=0)] * 5
        + [np.stack([mask_sw] + [mask_ss]*5 + [mask_se], axis=0)],
        axis=0
    )


def untile_patch(tiles: NDArray) -> NDArray:
    """ Stitches together 49 overlapping patches into a single patch.

    Args:
        patches (NDArray): A [7, 7, *, 80, 80] array of overlapping patches

    Returns:
        NDArray: A [*, 80, 80] patch
    """
    out = np.zeros(tiles.shape[2:-2] + (320, 320))
    weights = np.zeros_like(out)
    masks = get_tile_masks()
    stride = 40
    W = 80
    # stitching
    for i in range(7):
        for j in range(7):
            idx = np.index_exp[..., i * stride: i * stride + W, j * stride: j * stride + W]
            weights[idx] += masks[i, j]
            out[idx] += masks[i, j] * tiles[i, j]
    out /= weights
    # smoothing
    return out.reshape(-1, 80, 4, 80, 4).mean(axis=(-3, -1))
