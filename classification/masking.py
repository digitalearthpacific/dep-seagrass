from logging import Logger
from odc.stac import load
from pystac_client import Client
from xarray import DataArray, Dataset
from odc.algo import binary_dilation


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
        Dataset: Masked dataset, where 1 is land and 0 is water
    """
    land = (ds.mndwi).squeeze() < -0.2
    mask = land

    return apply_mask(ds, mask, ds_to_mask, return_mask)


# This is the updated mask_surf from mask_surf_refined_debug immersive
def mask_surf(
    ds: Dataset,
    ds_to_mask: Dataset | None = None,
    water_area_mask: DataArray | None = None,
    # --- FIX: Updated default thresholds to match notebook values ---
    surf_blue_threshold: float = 0.27,  # From notebook
    surf_green_threshold: float = 0.22,  # From notebook
    surf_red_threshold: float = 0.15,  # From notebook
    surf_nir_threshold: float = 0.08,  # From notebook
    surf_dilation_radius: int = 20,  # Notebook uses 40
    do_dilation: bool = True,
    return_mask: bool = False,
    # water_area_mask parameter removed - now derived internally
) -> Dataset:
    """Masks out surf / white water pixels based on multi-band reflectance
    and refines the mask by excluding land and applying morphological operations.

    Args:
        ds (Dataset): The input xarray Dataset containing the required spectral bands (blue, green, red, nir).
        ds_to_mask (Dataset | None, optional): The dataset to which the mask will be applied. If None, 'ds' will be masked. Defaults to None.
        surf_blue_threshold (float, optional): Blue reflectance threshold for surf.
        surf_green_threshold (float, optional): Green reflectance threshold for surf.
        surf_red_threshold (float, optional): Red reflectance threshold for surf.
        surf_nir_threshold (float, optional): NIR reflectance threshold for surf.
        surf_cleanup_radius (int, optional): Radius for initial erosion/dilation (opening) to remove small noise from surf mask. Defaults to 2.
        surf_dilation_radius (int, optional): Radius for the final binary dilation to expand the surf mask. Defaults to 40.
        return_mask (bool): If True, returns the binary surf mask. Otherwise, returns the masked dataset. Defaults to False.

    Returns:
        Dataset: The masked dataset, or the binary surf mask if 'return_mask' is True.
    """
    # 1. Create the initial raw surf mask based on multi-band high reflectance
    # This identifies very bright pixels that could be surf or bright sand/noise
    initial_surf_mask_raw = (
        (ds.blue > surf_blue_threshold)
        & (ds.green > surf_green_threshold)
        & (ds.red > surf_red_threshold)
        & (ds.nir > surf_nir_threshold)
    )

    # 2. Refine the raw surf mask to only include areas within the buffered water
    # This excludes bright land features and areas very close to land
    refined_surf_mask = initial_surf_mask_raw & water_area_mask

    if do_dilation:
        # 3. Perform the final binary dilation to create the desired buffer around surf areas
        rechunked = refined_surf_mask.chunk({"x": 512, "y": 512})
        refined_surf_mask = binary_dilation(
            rechunked, radius=surf_dilation_radius
        ).compute()

    # 4. Return the masked dataset or the final mask
    # The surf areas are to be ELIMINATED (masked out for seagrass habitats).
    return apply_mask(ds, refined_surf_mask, ds_to_mask, return_mask)


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
        Dataset: Masked dataset, which is 1 for deep water, 0 for not
    """
    mask = ds.ln_bg > threshold

    return apply_mask(ds, mask, ds_to_mask, return_mask)


def mask_elevation(
    ds: Dataset,
    ds_to_mask: Dataset | None = None,
    threshold: float = 10,
    return_mask: bool = False,
) -> Dataset:
    """
    Mask elevation. Returns 1 for high areas, 0 for low
    """
    e84_catalog = "https://earth-search.aws.element84.com/v1/"
    e84_client = Client.open(e84_catalog)
    collection = "cop-dem-glo-30"

    items = e84_client.search(
        collections=[collection], bbox=list(ds.odc.geobox.geographic_extent.boundingbox)
    ).item_collection()

    # Using geobox means it will load the elevation data the same shape as the other data
    elevation = load(items, measurements=["data"], geobox=ds.odc.geobox).squeeze()

    # True where data is above 10m elevation
    mask = elevation.data > threshold

    return apply_mask(ds, mask, ds_to_mask, return_mask)


def all_masks(
    ds: Dataset, return_mask: bool = False, log: Logger = Logger("mask")
) -> Dataset:
    log.info("Applying land mask...")
    _, land_mask = mask_land(ds, return_mask=True)

    log.info("Applying deeps mask...")
    _, deeps_mask = mask_deeps(ds, return_mask=True)

    log.info("Applying elevation mask...")
    _, elevation_mask = mask_elevation(ds, return_mask=True)

    # Pass the land_mask to mask_surf
    log.info("Applying surf mask...")
    _, surf_mask = mask_surf(
        ds=ds,
        water_area_mask=~land_mask,
        return_mask=True,
        # You can also pass surf_blue_threshold, surf_green_threshold, etc. here if you want to customize them
    )

    # Combine all masks. All individual masks are now False for areas to KEEP.
    log.info("Combining all masks...")
    mask = land_mask | deeps_mask | elevation_mask | surf_mask

    return apply_mask(ds, ~mask, None, return_mask)
