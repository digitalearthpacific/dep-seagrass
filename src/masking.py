from odc.stac import load
from pystac_client import Client
from xarray import DataArray, Dataset
from odc.algo import binary_dilation, mask_cleanup
from odc.algo import binary_dilation as oda_binary_dilation # Aliased import to prevent conflicts
from skimage.morphology import disk

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


# This is the updated mask_surf from mask_surf_refined_debug immersive
def mask_surf(
    ds: Dataset,
    ds_to_mask: Dataset | None = None,
    # --- FIX: Updated default thresholds to match notebook values ---
    surf_blue_threshold: float = 0.27, # From notebook
    surf_green_threshold: float = 0.22, # From notebook
    surf_red_threshold: float = 0.15, # From notebook
    surf_nir_threshold: float = 0.08, # From notebook
    # land_dilation_radius removed as it's handled by pre-processing water_area_mask
    surf_cleanup_radius: int = 2,   # Notebook uses 2
    surf_dilation_radius: int = 20, # Notebook uses 40
    return_mask: bool = False,
    water_area_mask: DataArray | None = None,
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
        land_dilation_radius (int): This parameter is now removed. The `water_area_mask` is expected to be pre-processed with land exclusion/buffering.
        surf_cleanup_radius (int, optional): Radius for initial erosion/dilation (opening) to remove small noise from surf mask. Defaults to 2.
        surf_dilation_radius (int, optional): Radius for the final binary dilation to expand the surf mask. Defaults to 40.
        return_mask (bool): If True, returns the binary surf mask. Otherwise, returns the masked dataset. Defaults to False.
        water_area_mask (DataArray | None, optional): A boolean DataArray (True for water, False for land/excluded areas). Used to constrain surf detection to water. Defaults to None.

    Returns:
        Dataset: The masked dataset, or the binary surf mask if 'return_mask' is True.
    """
    if water_area_mask is None:
        raise ValueError("water_area_mask must be provided to mask_surf for land exclusion.")

    # --- REMOVED LAND DILATION LOGIC FROM HERE ---
    # The water_area_mask passed in is now expected to be already buffered/processed
    # from mask_land's new logic (MNDWI + SWIR + potential land dilation in mask_land if needed there).
    buffered_water_area_mask = water_area_mask # Direct use of the provided water_area_mask


    # 1. Create the initial raw surf mask based on multi-band high reflectance
    # This identifies very bright pixels that could be surf or bright sand/noise
    initial_surf_mask_raw = (ds.blue > surf_blue_threshold) & \
                            (ds.green > surf_green_threshold) & \
                            (ds.red > surf_red_threshold) & \
                            (ds.nir > surf_nir_threshold)
    
    # Ensure raw mask is chunked and boolean
    initial_surf_mask_raw = initial_surf_mask_raw.chunk({'x': 512, 'y': 512}).astype(bool)

    # 2. Refine the raw surf mask to only include areas within the buffered water
    # This excludes bright land features and areas very close to land
    # --- FIX: Directly use the provided water_area_mask for refinement ---
    refined_surf_mask = initial_surf_mask_raw & buffered_water_area_mask
    
    # Ensure it's chunked and boolean
    refined_surf_mask = refined_surf_mask.chunk({'x': 512, 'y': 512}).astype(bool)

    # 3. Apply mask_cleanup (erosion then dilation) to remove small noise specks from the refined surf mask
    # This is an "opening" operation.
    # cleaned_surf_mask = mask_cleanup(refined_surf_mask, [["erosion", surf_cleanup_radius], ["dilation", surf_cleanup_radius]])
    
    # Ensure output of cleanup is chunked and boolean for final dilation
    cleaned_surf_mask = refined_surf_mask.chunk({'x': 512, 'y': 512}).astype(bool)

    # 4. Perform the final binary dilation to create the desired buffer around surf areas
    final_dilated_surf_mask = oda_binary_dilation(cleaned_surf_mask, radius=surf_dilation_radius)

    # 5. Return the masked dataset or the final mask
    # The surf areas are to be ELIMINATED (masked out for seagrass habitats).
    # So, the mask to return for `apply_mask` for *keeping* desired areas should be `~final_dilated_surf_mask`.
    mask_to_return_for_keeping = ~final_dilated_surf_mask

    return apply_mask(ds, mask_to_return_for_keeping, ds_to_mask, return_mask)


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
    print("Applying land mask...")
    # Get the water_area_mask (True for WATER, False for LAND/SWIR_BRIGHT)
    _, water_area_mask = mask_land(ds, return_mask = True)
    
    print("Applying deeps mask...")
    _, deeps_mask = mask_deeps(ds, return_mask = True)
    print("Applying elevation mask...")
    _, elevation_mask = mask_elevation(ds, return_mask = True)
    
    # Pass the water_area_mask to mask_surf
    # mask_surf will return a mask that is True for NON-SURF areas (areas to KEEP)
    print("Applying surf mask...")
    _, surf_mask_for_keeping = mask_surf(
        ds=ds,
        return_mask = True, 
        water_area_mask = water_area_mask # <--- Passing the correct water_area_mask
        # You can also pass surf_blue_threshold, surf_green_threshold, etc. here if you want to customize them
    )
    
    # Combine all masks. All individual masks are now True for areas to KEEP.
    print("Combining all masks...")
    # Initialize the combined mask with the first mask (`water_area_mask`)
    # and then combine it with subsequent masks using logical AND (`&`).
    mask = water_area_mask & deeps_mask & elevation_mask & surf_mask_for_keeping

    return apply_mask(ds, mask, None, return_mask)