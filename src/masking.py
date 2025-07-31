from odc.stac import load
from pystac_client import Client
from xarray import DataArray, Dataset

def apply_mask(
    ds: Dataset,
    mask: DataArray,
    ds_to_mask: Dataset | None = None,
    return_mask: bool = False,
) -> Dataset:
    """Applies a mask to a dataset"""
    to_mask = ds if ds_to_mask is None else ds_to_mask
    masked = to_mask.where(mask)

    if return_mask:
        return masked, mask
    else:
        return masked


def mask_land(
    ds: Dataset, ds_to_mask: Dataset | None = None, return_mask: bool = False
) -> Dataset:
    """Masks out land pixels based on the NDWI and MNDWI indices.

    Args:
        ds (Dataset): Dataset to mask
        ds_to_mask (Dataset | None, optional): Dataset to mask. Defaults to None.
        return_mask (bool, optional): If True, returns the mask as well. Defaults to False.

    Returns:
        Dataset: Masked dataset
    """
    land = (ds.mndwi).squeeze() < -0.2
    # mask = mask_cleanup(land, [["dilation", 5], ["erosion", 5]])
    mask = land

    # Inverting the mask here
    mask = ~mask

    return apply_mask(ds, mask, ds_to_mask, return_mask)


def mask_surf(
    ds: Dataset,
    ds_to_mask: Dataset | None = None,
    threshold: float = 0.001,
    return_mask: bool = False,
) -> Dataset:
    """Masks out surf / white water pixels based on the nir

    Args:
        ds (Dataset): Dataset to mask
        ds_to_mask (Dataset | None, optional): Dataset to mask. Defaults to None.
        threshold (float, optional): Threshold for the natural log of the blue/green. Defaults to 0.2.
        return_mask (bool, optional): If True, returns the mask as well. Defaults to False.

    Returns:
        Dataset: Masked dataset
    """
    mask = ds.nir > threshold

    return apply_mask(ds, mask, ds_to_mask, return_mask)


def mask_deeps(
    ds: Dataset,
    ds_to_mask: Dataset | None = None,
    threshold: float = -0.08,
    return_mask: bool = False,
) -> Dataset:
    """Masks out deep water pixels based on the natural log of the blue/green

    Args:
        ds (Dataset): Dataset to mask
        ds_to_mask (Dataset | None, optional): Dataset to mask. Defaults to None.
        threshold (float, optional): Threshold for the natural log of the blue/green. Defaults to 0.2.
        return_mask (bool, optional): If True, returns the mask as well. Defaults to False.

    Returns:
        Dataset: Masked dataset
    """
    mask = ds.ln_bg < threshold
    # mask = mask_cleanup(mask, [["erosion", 5], ["dilation", 5]])

    return apply_mask(ds, mask, ds_to_mask, return_mask)


def mask_elevation(
    ds: Dataset,
    ds_to_mask: Dataset | None = None,
    threshold: float = 10,
    return_mask: bool = False,
) -> Dataset:

    e84_catalog = "https://earth-search.aws.element84.com/v1/"
    e84_client = Client.open(e84_catalog)
    collection = "cop-dem-glo-30"

    items = e84_client.search(
        collections=[collection], bbox=list(ds.odc.geobox.geographic_extent.boundingbox)
    ).item_collection()

    # Using geobox means it will load the elevation data the same shape as the other data
    elevation = load(items, measurements=["data"], geobox=ds.odc.geobox).squeeze()

    # True where data is above 10m elevation
    mask = elevation.data < threshold

    return apply_mask(ds, mask, ds_to_mask, return_mask)


def all_masks(
    ds: Dataset,
    return_mask: bool = False,
) -> Dataset:
    _, land_mask = mask_land(ds, return_mask=True)
    _, deeps_mask = mask_deeps(ds, return_mask=True)
    _, surf_mask = mask_surf(ds, return_mask=True)
    _, elevation_mask = mask_elevation(ds, return_mask=True)

    mask = land_mask & deeps_mask & surf_mask & elevation_mask

    return apply_mask(ds, mask, None, return_mask)
