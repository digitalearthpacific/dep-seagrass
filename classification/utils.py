from odc.stac import load  # Correct source for `load`
import xarray as xr
from xarray import DataArray, Dataset
import numpy as np
from odc.algo import mask_cleanup
from skimage.feature import graycomatrix, graycoprops
from skimage import data
from skimage.util import view_as_windows
from shapely import box
from datetime import datetime
from shapely.geometry import Polygon
from pyproj import CRS
import folium
import geopandas as gpd
import pandas as pd
import rasterio as rio
import rioxarray
from ipyleaflet import basemaps
from numpy.lib.stride_tricks import sliding_window_view
import pystac_client
import planetary_computer
from odc.stac import load
from pystac.client import Client
from skimage.feature import graycomatrix, graycoprops
from odc.algo import binary_dilation
from skimage.morphology import disk
from scipy.ndimage import binary_dilation


def load_data(items, bands, bbox):
    """
    Load data into a dataset with specified measurements and configurations.

    Parameters:
    - items: List of STAC items to load.
    - bbox: Bounding box for the region of interest.

    Returns:
    - data: The loaded dataset.
    """
    data = load(
        items,
        bands=[
            "nir",
            "red",
            "blue",
            "green",
            "emad",
            "smad",
            "bcmad",
            # "count",
            "green",
            "nir08",
            "nir09",
            "swir16",
            "swir22",
            "coastal",
            "rededge1",
            "rededge2",
            "rededge3",
        ],
        bbox=bbox,
        chunks={"x": 2048, "y": 2048},
        groupby="solar_day",
    )
    return data


def calculate_band_indices(data):
    """
    Calculate various band indices and add them to the dataset.

    Parameters:
    data (xarray.Dataset): The input dataset containing the necessary spectral bands.

    Returns:
    xarray.Dataset: The dataset with added band indices.
    """

    data["mndwi"] = (data["green"] - data["swir16"]) / (data["green"] + data["swir16"])
    data["ndti"] = (data["red"] - data["green"]) / (data["red"] + data["green"])
    data["cai"] = (data["coastal"] - data["blue"]) / (data["coastal"] + data["blue"])
    data["ndvi"] = (data["nir"] - data["red"]) / (data["nir"] + data["red"])
    data["nir"] + (6 * data["red"]) - (7.5 * data["blue"] + 1)
    data["evi"] = (2.5 * data["nir"] - data["red"]) / (
        data["nir"] + (6 * data["red"]) - (7.5 * data["blue"]) + 1
    )
    data["savi"] = (data["nir"] - data["red"]) / (data["nir"] + data["red"])
    data["ndwi"] = (data["green"] - data["nir"]) / (data["green"] + data["nir"])
    data["b_g"] = data["blue"] / data["green"]
    data["b_r"] = data["blue"] / data["red"]
    data["mci"] = data["nir"] / data["rededge1"]
    data["ndci"] = (data["rededge1"] - data["red"]) / (data["rededge1"] + data["red"])
    data["ln_bg"] = np.log(data.blue / data.green)

    return data


def scale(data):
    """
    Scale the input data by applying a factor and clipping the values.

    Parameters:
    data (xr.Dataset): The input dataset containing the bands to be scaled.

    Returns:
    xr.Dataset: The scaled dataset with values clipped between 0 and 1.
    """
    scaled = (data * 0.0001).clip(0, 1)
    return scaled


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


# def mask_surf(
#     ds: Dataset,
#     ds_to_mask: Dataset | None = None,
#     threshold: float = 0.001,
#     return_mask: bool = False,
#     dilation_radius: int = 20, # <--- NEW PARAMETER ADDED HERE
# ) -> Dataset:
#     """Masks out surf / white water pixels based on the nir

#     Args:
#         ds (Dataset): Dataset to mask
#         ds_to_mask (Dataset | None, optional): Dataset to mask. Defaults to None.
#         threshold (float, optional): Threshold for the natural log of the blue/green. Defaults to 0.001.
#         dilation_radius (int, optional): Radius for binary dilation to expand the mask. Defaults to 20.
#         return_mask (bool, optional): If True, returns the mask as well. Defaults to False.

#     Returns:
#         Dataset: Masked dataset
#     """
#     mask = ds.nir > threshold
#     mask = mask.chunk({'x': 512, 'y': 512}) # Keep this if it's necessary for your Dask workflow

#     # Convert to boolean before dilation to ensure it's treated as a binary mask
#     mask_bool = mask.astype(bool)

#     # Pass the new dilation_radius parameter to binary_dilation
#     dilated_mask = binary_dilation(mask_bool, radius=dilation_radius) # <--- USE THE PARAMETER HERE

#     return apply_mask(ds, dilated_mask, ds_to_mask, return_mask)


# def mask_surf(
#     ds: Dataset,
#     ds_to_mask: Dataset | None = None,
#     threshold: float = 0.001,
#     return_mask: bool = False,
# ) -> Dataset:
#     """Masks out surf / white water pixels based on the nir

#     Args:
#         ds (Dataset): Dataset to mask
#         ds_to_mask (Dataset | None, optional): Dataset to mask. Defaults to None.
#         threshold (float, optional): Threshold for the NIR value to identify surf. Defaults to 0.001.
#         return_mask (bool, optional): If True, returns the mask as well. Defaults to False.

#     Returns:
#         Dataset: Masked dataset
#     """
# from scipy.ndimage import binary_dilation, generate_binary_structure
# # If you prefer to use skimage.morphology.disk, ensure it's imported:
# # from skimage.morphology import disk
# import numpy as np # Needed for np.ones for structuring element fallback

# # Assuming Dataset and apply_mask are defined elsewhere in your utils.py
# # (They were in your previous full utils.py immersive)

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
        threshold (float, optional): Threshold for the NIR value to identify surf. Defaults to 0.001.
        return_mask (bool, optional): If True, returns the mask as well. Defaults to False.

    Returns:
        Dataset: Masked dataset
    """
    # 1. Create the initial binary mask for surf
    initial_mask = ds.nir > threshold

    # 2. Chunking is appropriate for Dask arrays
    initial_mask = initial_mask.chunk({'x': 512, 'y': 512})

    # 3. Define the structuring element for binary_dilation
    # For scipy.ndimage.binary_dilation, 'radius' is not a direct argument.
    # Instead, you create a structuring element (e.g., a disk/sphere).

    dilation_radius = 40 # The desired dilation radius in pixels

    try:
        # Attempt to use skimage.morphology.disk for a circular structuring element
        # This is generally preferred for creating disks for dilation
        from skimage.morphology import disk
        structuring_element = disk(dilation_radius)
    except ImportError:
        # Fallback if scikit-image is not installed or disk is unavailable
        # generate_binary_structure creates a connectivity-based structure, not strictly a disk.
        # For a large 'radius', generate_binary_structure can create very large, sparse arrays.
        # A simpler approach for approximate large radius with scipy might be a large square:
        print("Warning: skimage.morphology.disk not found. Using approximate square structuring element.")
        structuring_element = np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1), dtype=bool)

    # 4. Perform the binary dilation using the 'structure' argument
    # Ensure initial_mask is computed or handled by Dask correctly before dilation
    # It's good practice to compute small masks if they fit in memory before passing to scipy.ndimage functions
    # For potentially large masks, scipy.ndimage might operate better if they are NumPy arrays.
    # If initial_mask is an xarray DataArray with Dask chunks, binary_dilation might implicitly compute.
    # For explicit computation if needed: initial_mask.compute().values
    print(f"DEBUG: initial_mask min: {initial_mask.min().compute()}, max: {initial_mask.max().compute()}")
    print(f"DEBUG: initial_mask sum (True pixels): {initial_mask.sum().compute()}")
    # Optionally, save or plot the initial mask for visual inspection
    # initial_mask.plot(figsize=(8,8), cmap='gray') # Requires matplotlib
    # plt.show()
    
    # Pass the structuring_element as the 'structure' argument
    expanded_mask = binary_dilation(initial_mask, structure=structuring_element)

    # 5. Crucially: Return the *expanded_mask*, not the original initial_mask
    return apply_mask(ds, expanded_mask, ds_to_mask, return_mask)

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


def do_prediction(ds, model, output_name: str | None = None):
    """Predicts the model on the dataset and adds the prediction as a new variable.

    Args:
        ds (Dataset): Dataset to predict on
        model (RegressorMixin): Model to predict with
        output_name (str | None): Name of the output variable. Defaults to None.

    Returns:
        Dataset: Dataset with the prediction as a new variable
    """
    mask = ds.red.isnull()  # Probably should check more bands

    # Convert to a stacked array of observations
    stacked_arrays = ds.to_array().stack(dims=["y", "x"])

    # Replace any infinities with NaN
    stacked_arrays = stacked_arrays.where(stacked_arrays != float("inf"))
    stacked_arrays = stacked_arrays.where(stacked_arrays != float("-inf"))

    # Replace any NaN values with 0
    # TODO: Make sure that each column is labelled with the correct band name
    stacked_arrays = stacked_arrays.squeeze().fillna(0).transpose()

    # Predict the classes
    predicted = model.predict(stacked_arrays)

    # Reshape back to the original 2D array
    array = predicted.reshape(ds.y.size, ds.x.size)

    # Convert to an xarray again, because it's easier to work with
    predicted_da = xr.DataArray(array, coords={"y": ds.y, "x": ds.x}, dims=["y", "x"])

    # Mask the prediction with the original mask
    # predicted_da = predicted_da.where(~mask).compute()

    # If we have a name, return dataset, else the dataarray
    if output_name is None:
        return predicted_da
    else:
        return predicted_da.to_dataset(name=output_name)


WINDOW_SIZE = 9
LEVELS = 32


# Your patch function
def glcm_features(patch):
    glcm = graycomatrix(
        patch, distances=[1], angles=[0], levels=LEVELS, symmetric=True, normed=True
    )
    out = np.empty(7, dtype=np.float32)
    out[0] = graycoprops(glcm, "contrast")[0, 0]
    out[1] = graycoprops(glcm, "homogeneity")[0, 0]
    out[2] = graycoprops(glcm, "energy")[0, 0]
    out[3] = graycoprops(glcm, "ASM")[0, 0]
    out[4] = graycoprops(glcm, "correlation")[0, 0]
    out[5] = graycoprops(glcm, "mean")[0, 0]

    # glcm_p = glcm[:, :, 0, 0]
    # entropy[i, j] = -np.sum(glcm_p * np.log2(glcm_p + 1e-10))

    glcm_p = glcm[:, :, 0, 0]
    out[6] = -np.sum(glcm_p * np.log2(glcm_p + 1e-10))
    return out


def output(
    classification_da: xr.DataArray,
    target_class_id: int,
    output_dtype: str = "float32",
    nodata_value: int = 255,  # Value to use for NoData if output_dtype is an integer type
) -> xr.DataArray:
    """ """

    # 1. Boolean mask for the target class
    target_class = classification_da == target_class_id

    # 2. Boolean mask for all non-NaN (classified) pixels
    is_classified = classification_da.notnull()

    # 3. Handle NaN.
    binary_output_float = xr.full_like(classification_da, np.nan, dtype=float)

    # 4. Set target class to 1.0 where target_class is True
    binary_output_float = xr.where(target_class, 1.0, binary_output_float)

    # 5. Set other classified areas to 0.0
    #    Other marine benethic habitats but not NaNs.
    binary_output_float = xr.where(
        is_classified & ~target_class, 0.0, binary_output_float
    )

    # 6. Handle final dtype conversion and NoData value
    if output_dtype in ["uint8", "int8", "int16", "int32", "int64"]:
        # Check if the nodata_value is valid for the chosen dtype
        dtype_info = np.iinfo(output_dtype)
        if not (dtype_info.min <= nodata_value <= dtype_info.max):
            print(
                f"Warning: `nodata_value` ({nodata_value}) is outside the valid range "
                f"for `output_dtype` ({output_dtype}) which is [{dtype_info.min}, {dtype_info.max}]. "
                f"This might lead to unexpected results or data wrapping."
            )

        print(
            f"Casting to {output_dtype} and converting NaN (original no data) values to {nodata_value}."
        )
        # Replace NaNs with the specified nodata_value
        final_output = binary_output_float.fillna(nodata_value).astype(output_dtype)
        # Add nodata attributes for GeoTIFF writing, crucial for GIS software
        final_output.attrs["_FillValue"] = nodata_value
        final_output.attrs["nodata"] = (
            nodata_value  # Common attribute for geospatial data
        )
    else:  # For float types, NaN is the natural nodata
        final_output = binary_output_float.astype(output_dtype)
        # Ensure _FillValue is set if there's an original nodata value in the input
        if "_FillValue" in classification_da.attrs:
            final_output.attrs["_FillValue"] = classification_da.attrs["_FillValue"]
        elif "nodata" in classification_da.attrs:
            final_output.attrs["nodata"] = classification_da.attrs["nodata"]

    return final_output


def probability(
    ds: xr.Dataset,
    model,  # Your trained scikit-learn model (e.g., RandomForestClassifier)
    bands: list[str],  # List of band names to use as features
    target_class_id: int,  # The class ID for which to return probabilities
    no_data_value: int = -9999,  # Value used for no-data in your input data (if applicable)
    scale_to_100: bool = True,  # Whether to scale probabilities from 0-1 to 0-100
) -> xr.DataArray:
    """
    Generates a probability raster for a specific target class from a trained
    Random Forest classifier.

    - Pixels that were originally NoData (NaN or `no_data_value`) will remain NaN in the output.
    - Probabilities will range from 0.0 to 1.0 (or 0 to 100 if `scale_to_100` is True).

    Parameters:
    - ds (xr.Dataset): The input xarray Dataset containing the bands for prediction.
                       Expected to have spatial dimensions (e.g., 'x', 'y').
    - model: The trained scikit-learn classifier model.
    - bands (list[str]): A list of string names of the bands/variables in `ds`
                         to be used as features for prediction.
    - target_class_id (int): The integer ID of the class for which to generate
                             the probability raster (e.g., 4 for seagrass).
    - no_data_value (int): The integer value representing no-data in the input bands.
                           Pixels with this value will be excluded from prediction
                           and remain as NaN in the output.
    - scale_to_100 (bool): If True, probabilities will be scaled from 0-1 to 0-100.

    Returns:
    - xr.DataArray: An xarray DataArray containing the probability for the
                    `target_class_id`, with original spatial dimensions and georeferencing.
                    Values are floats.

    Raises:
    - ValueError: If the `target_class_id` is not found in the model's classes,
                  or if the number of input features (bands) does not match
                  what the model was trained on.
    - TypeError: If `ds` is not an xarray Dataset.
    """

    if not isinstance(ds, xr.Dataset):
        raise TypeError("Input 'ds' must be an xarray.Dataset.")
    if target_class_id not in model.classes_:
        raise ValueError(
            f"Target class ID {target_class_id} not found in model classes: {model.classes_}"
        )

    # print(f"Generating probability raster for class ID: {target_class_id}")

    # --- DEBUGGING START ---
    # print(f"DEBUG: 'bands' list provided to function: {bands}")
    # print(f"DEBUG: Length of 'bands' list provided: {len(bands)}")
    # --- DEBUGGING END ---

    # --- NEW FIX: Ensure spatial coordinates are numeric (float) ---
    # Create a copy of the dataset to avoid modifying the original in-place
    # This also ensures that if 'time' is a dimension, it's handled correctly
    # by ensuring its coordinate is numeric if present.
    ds_copy = ds.copy()

    # Check and convert 'y' coordinate
    if "y" in ds_copy.coords and ds_copy["y"].dtype.kind not in [
        "i",
        "f",
    ]:  # if not integer or float
        print(f"DEBUG: Converting 'y' coordinate from {ds_copy['y'].dtype} to float.")
        ds_copy["y"] = ds_copy["y"].astype(float)

    # Check and convert 'x' coordinate
    if "x" in ds_copy.coords and ds_copy["x"].dtype.kind not in [
        "i",
        "f",
    ]:  # if not integer or float
        print(f"DEBUG: Converting 'x' coordinate from {ds_copy['x'].dtype} to float.")
        ds_copy["x"] = ds_copy["x"].astype(float)

    # Check and convert 'time' coordinate if it exists and is not numeric/datetime
    if "time" in ds_copy.coords and ds_copy["time"].dtype.kind not in [
        "i",
        "f",
        "M",
    ]:  # M for datetime64
        print(
            f"DEBUG: Converting 'time' coordinate from {ds_copy['time'].dtype} to datetime64[ns]."
        )
        # Use pandas.to_datetime for robust conversion of time-like strings/objects
        ds_copy["time"] = pd.to_datetime(ds_copy["time"].values)
    # --- END NEW FIX ---

    # 1. Select the relevant bands and stack spatial dimensions
    features_ds = ds_copy[
        bands
    ]  # Selects only the specified bands from the potentially modified copy

    # Check if 'time' is a dimension in features_ds. If so, stack it as well
    # to flatten all samples across time and space.
    stack_dims = ["y", "x"]
    if "time" in features_ds.dims:
        stack_dims.insert(0, "time")  # Add time as the first dimension to stack

    features_stacked = features_ds.stack(pixel=stack_dims)

    # 2. Convert to a Pandas DataFrame
    features_df = features_stacked.to_dataframe()

    # --- NEW FIX: Explicitly drop coordinate columns that might appear as data columns ---
    # This addresses the issue where to_dataframe() might promote index levels
    # (like y, x, time, spatial_ref) to regular columns if they are also coordinates.
    # We only want the actual feature bands.
    # Only drop columns that are actually present in the DataFrame's columns
    columns_to_drop = [
        col for col in ["y", "x", "time", "spatial_ref"] if col in features_df.columns
    ]
    if columns_to_drop:
        features_df = features_df.drop(columns=columns_to_drop)
    # --- END NEW FIX ---

    # --- DEBUGGING START ---
    # print(f"DEBUG: Columns of features_df (after dropping coordinates, before NaNs/no_data): {features_df.columns.tolist()}")
    # print(f"DEBUG: Number of columns in features_df (after dropping coordinates): {len(features_df.columns)}")
    # --- DEBUGGING END ---

    # Store the original index (pixel coordinates) before dropping NaNs/no_data_value
    original_pixel_index = features_df.index

    # Ensure all features are numeric before dropping NaNs
    features_for_prediction_df = features_df.apply(pd.to_numeric, errors="coerce")
    features_for_prediction_df = features_for_prediction_df.dropna()

    # Filter out rows where all feature values are equal to no_data_value
    if no_data_value is not None and not np.isnan(no_data_value):
        # Create a boolean mask where at least one feature is NOT the no_data_value
        # This handles cases where some bands might have valid data while others are nodata
        # For simplicity, we assume if ANY band is no_data_value, the pixel is nodata.
        is_not_nodata_mask = (features_for_prediction_df != no_data_value).any(axis=1)
        features_for_prediction_df = features_for_prediction_df[is_not_nodata_mask]

    # Store the index of the valid (non-NaN, non-nodata) pixels that will be predicted
    valid_pixel_index = features_for_prediction_df.index

    # Convert the DataFrame to a NumPy array for sklearn
    features_for_prediction_np = features_for_prediction_df.values

    # --- NEW CHECK FOR FEATURE COUNT ---
    expected_features = model.n_features_in_
    actual_features = features_for_prediction_np.shape[1]

    # --- DEBUGGING START ---
    # print(f"DEBUG: Shape of features_for_prediction_np (after processing): {features_for_prediction_np.shape}")
    # print(f"DEBUG: Model expects {expected_features} features (model.n_features_in_).")
    # print(f"DEBUG: Actual features in input data (from features_for_prediction_np): {actual_features}.")
    # --- DEBUGGING END ---

    if actual_features != expected_features:
        raise ValueError(
            f"Feature count mismatch: Input data has {actual_features} features "
            f"but the model expects {expected_features} features. "
            f"Please ensure the 'bands' list matches the features used during model training."
        )
    # --- END NEW CHECK ---

    # 3. Get probabilities for all classes
    probabilities_np = model.predict_proba(features_for_prediction_np)

    # Find the index of the target class in the model's output
    target_class_index = list(model.classes_).index(target_class_id)

    # Extract probabilities for the target class
    target_class_probabilities_1d = probabilities_np[:, target_class_index]

    # --- NEW FIX: Map probabilities back using unstacking ---
    # Create a temporary DataArray with the probabilities and the valid pixel index
    temp_prob_1d_da = xr.DataArray(
        target_class_probabilities_1d,
        coords={"pixel": valid_pixel_index},  # Use the MultiIndex as the coordinate
        dims=["pixel"],
    )

    # Unstack the 'pixel' dimension back into original spatial dimensions (e.g., 'y', 'x' or 'time', 'y', 'x')
    # This creates a DataArray with probabilities in the correct spatial locations.
    # It will have NaNs where pixels were originally dropped.
    # The dimensions for unstacking should match those used in features_stacked.stack()
    unstacked_prob_da = temp_prob_1d_da.unstack("pixel")

    # Create the final probability_da with the full original grid from ds_copy[bands[0]]
    # and then update it with the unstacked probabilities.
    # This ensures that areas that were originally masked out but not part of valid_pixel_index
    # (e.g., outside the original data extent or areas that were purely NaN) remain NaN.
    # Use ds_copy[bands[0]] as the template, as it has the correct original dimensions and coordinates
    # (and its coordinates have been converted to numeric types if needed).
    probability_da = xr.full_like(ds_copy[bands[0]], np.nan, dtype=float)
    probability_da.name = f"probability_class_{target_class_id}"

    # Combine the full NaN template with the unstacked probabilities.
    # This will fill in the computed probabilities where they exist,
    # and leave NaNs elsewhere (original no-data areas).
    probability_da = probability_da.combine_first(unstacked_prob_da)
    # --- END NEW FIX ---

    # 5. Scale to 0-100 if requested
    if scale_to_100:
        probability_da = probability_da * 100

    # 6. Copy georeferencing attributes from a source band
    # This assumes the first band in `bands` list (`ds[bands[0]]`) has correct georeferencing
    # Copy from the original `ds` to preserve any original attributes if `ds_copy` was simplified
    probability_da.attrs = ds[bands[0]].attrs
    # Ensure CRS is maintained if using rioxarray
    # if hasattr(ds[bands[0]], 'rio'):
    #     probability_da = probability_da.rio.write_crs(ds[bands[0]].rio.crs)
    #     probability_da = probability_da.rio.write_transform(ds[bands[0]].rio.transform())

    print("Probability raster generation complete.")
    return probability_da


def proba_binary(
    probability_da: xr.DataArray,
    threshold: float,  # Threshold value (e.g., 60 for 80%)
    output_dtype: str = "uint8",
    nodata_value: int = 255,  # Value to use for NoData if output_dtype is an integer type
) -> xr.DataArray:
    """
    Converts a probability raster into a binary classification raster based on a threshold.

    - Pixels with probability >= threshold are set to 1.
    - Pixels with probability < threshold (but are valid data) are set to 0.
    - Pixels that were originally NoData (NaN) remain NoData (converted to `nodata_value`
      if `output_dtype` is an integer type).

    Parameters:
    - probability_da (xr.DataArray): Input DataArray with probability values (e.g., 0-100).
                                    Expected to have spatial dimensions (e.g., 'x', 'y').
    - threshold (float): The threshold value to apply. Pixels with probability >= threshold
                         will be classified as 1.
    - output_dtype (str): The desired output data type. Use 'float32' or 'float64'
                          to preserve NaN values. If an integer type (e.g., 'uint8', 'int16'),
                          original NaNs will be converted to `nodata_value`.
    - nodata_value (int): The value to use for NoData if output_dtype is an integer type.
                          Must be within the range of the chosen integer `output_dtype`.
                          Default is 255.

    Returns:
    - xr.DataArray: A new DataArray with binary classification (1 for above threshold,
                    0 for below threshold, and `nodata_value` for NoData areas).
    """

    print(f"Thresholding probability raster at {threshold}% to binary output.")

    # 1. Boolean mask for pixels above or equal to the threshold
    is_above_threshold = probability_da >= threshold

    # 2. Boolean mask for all non-NaN (valid) pixels
    is_valid_data = probability_da.notnull()

    # 3. Initialize the output array with NaN (for original NoData areas)
    #    We use float type for intermediate calculations to handle NaN.
    binary_output_float = xr.full_like(probability_da, np.nan, dtype=float)

    # 4. Set pixels above threshold to 1.0
    binary_output_float = xr.where(is_above_threshold, 1.0, binary_output_float)

    # 5. Set pixels below threshold (but valid data) to 0.0
    #    This applies where it's valid data AND below the threshold.
    binary_output_float = xr.where(
        is_valid_data & ~is_above_threshold, 0.0, binary_output_float
    )

    # 6. Handle final dtype conversion and NoData value
    if output_dtype in ["uint8", "int8", "int16", "int32", "int64"]:
        # Check if the nodata_value is valid for the chosen dtype
        dtype_info = np.iinfo(output_dtype)
        if not (dtype_info.min <= nodata_value <= dtype_info.max):
            print(
                f"Warning: `nodata_value` ({nodata_value}) is outside the valid range "
                f"for `output_dtype` ({output_dtype}) which is [{dtype_info.min}, {dtype_info.max}]. "
                f"This might lead to unexpected results or data wrapping."
            )

        print(
            f"Casting to {output_dtype} and converting NaN (original no data) values to {nodata_value}."
        )
        # Replace NaNs with the specified nodata_value before casting to integer type
        final_output = binary_output_float.fillna(nodata_value).astype(output_dtype)
        # Add nodata attributes for GeoTIFF writing, crucial for GIS software
        final_output.attrs["_FillValue"] = nodata_value
        final_output.attrs["nodata"] = (
            nodata_value  # Common attribute for geospatial data
        )
    else:  # For float types, NaN is the natural nodata
        final_output = binary_output_float.astype(output_dtype)
        # Ensure _FillValue is set if there's an original nodata value in the input
        if "_FillValue" in probability_da.attrs:
            final_output.attrs["_FillValue"] = probability_da.attrs["_FillValue"]
        elif "nodata" in probability_da.attrs:
            final_output.attrs["nodata"] = probability_da.attrs["nodata"]

    return final_output


# import geopandas as gpd
import os
import pandas as pd  # Explicitly import pandas as it's used by geopandas.pd.concat


def standardise(
    input_folder_path: str,
    output_folder_path: str,
    file_extension: str = ".geojson",
    target_crs: str = "EPSG:4326",
    category_to_id_map: dict = None,
    id_to_category_map: dict = None,
    default_unmapped_id: int = 0,
    output_file_format: str = "geojson",  # New parameter for output format
    **kwargs,  # Allows passing extra arguments to gpd.read_file and gpd.to_file
) -> None:
    """
    Loads spatial files from an input folder, standardises their schema (observed, cc_id),
    remaps categories (e.g., seaweed to algae), ensures each is in the target CRS,
    and then saves each processed GeoDataFrame to an output folder.

    Parameters:
    - input_folder_path (str): The path to the folder containing the original spatial files.
    - output_folder_path (str): The path to the folder where processed files will be saved.
                                 This folder will be created if it doesn't exist.
    - file_extension (str): The file extension to filter for when reading files
                            (e.g., ".geojson", ".shp", ".csv").
    - target_crs (str, optional): The CRS to ensure each individual GeoDataFrame is in.
                                  Defaults to "EPSG:4326" (WGS84 Lat/Lon).
                                  Set to None to skip any CRS handling.
    - category_to_id_map (dict, optional): A dictionary mapping string category names to integer IDs.
                                           Used to create/update 'cc_id'.
    - id_to_category_map (dict, optional): A dictionary mapping integer IDs to string category names.
                                           Used to create/update 'observed'.
    - default_unmapped_id (int, optional): The integer ID to assign to 'cc_id' for 'observed'
                                           categories not found in `category_to_id_map`. Defaults to 0.
    - output_file_format (str, optional): The format to save the output files.
                                          Common options: "geojson", "shp", "csv". Defaults to "geojson".
    - **kwargs: Additional keyword arguments to pass to gpd.read_file() (for reading)
                and gpd.to_file() (for writing).
                For example, for reading CSVs: `sep=','`, `on_bad_lines='skip'`.
                For writing GeoJSONs: `driver='GeoJSON'`.
    """
    print(
        f"Starting standardization and resaving process from '{input_folder_path}' to '{output_folder_path}'."
    )

    # Create output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)
    print(f"Ensuring output directory exists: {output_folder_path}")

    if target_crs:
        print(f"All processed GeoDataFrames will be ensured to be in CRS: {target_crs}")

    if not os.path.exists(input_folder_path):
        print(f"Error: Input folder '{input_folder_path}' does not exist. Aborting.")
        return

    processed_count = 0
    skipped_count = 0

    # List all files in the input folder
    for filename in os.listdir(input_folder_path):
        file_path = os.path.join(input_folder_path, filename)

        if os.path.isfile(file_path) and filename.endswith(file_extension):
            try:
                gdf = gpd.read_file(file_path, **kwargs)

                print(f"\nProcessing: {filename}")

                # --- Schema and Category Handling ---

                # 1. standardise category column name to 'observed'
                found_category_col = None
                for col in gdf.columns:
                    if col.lower() in [
                        "observed",
                        "category",
                        "class_name",
                        "classname",
                        "label",
                    ]:
                        found_category_col = col
                        break

                if found_category_col and found_category_col != "observed":
                    gdf = gdf.rename(columns={found_category_col: "observed"})
                    print(f"  Renamed '{found_category_col}' to 'observed'.")
                elif (
                    not found_category_col
                    and "cc_id" in gdf.columns
                    and id_to_category_map
                ):
                    gdf["observed"] = gdf["cc_id"].map(id_to_category_map)
                    print(f"  Created 'observed' from 'cc_id'.")
                elif not found_category_col:
                    print(
                        f"  Warning: No 'observed' or similar category column found. "
                        "Attempting to proceed, but 'observed' column might be missing or incorrect."
                    )

                # 2. Rename 'seaweed' to 'algae' in the 'observed' column
                if "observed" in gdf.columns:
                    gdf["observed"] = gdf["observed"].astype(str)
                    original_seaweed_count = (gdf["observed"] == "seaweed").sum()
                    if original_seaweed_count > 0:
                        gdf["observed"] = gdf["observed"].replace("seaweed", "algae")
                        print(
                            f"  Renamed 'seaweed' to 'algae' ({original_seaweed_count} instances)."
                        )

                # 3. Ensure 'cc_id' column based on 'observed' and canonical mapping
                if "observed" in gdf.columns and category_to_id_map:
                    gdf["cc_id"] = (
                        gdf["observed"]
                        .map(category_to_id_map)
                        .fillna(default_unmapped_id)
                        .astype(int)
                    )
                    print(f"  Ensured 'cc_id' column.")
                elif "cc_id" not in gdf.columns:
                    print(f"  Warning: No 'cc_id' column could be created or found.")

                # --- Robust CRS Handling ---
                if target_crs:
                    if gdf.crs is None:
                        print(
                            f"  Warning: File has no CRS. Assuming it's in {target_crs} and setting CRS."
                        )
                        gdf = gdf.set_crs(target_crs, allow_override=True)

                    if gdf.crs is not None and gdf.crs != target_crs:
                        print(f"  Reprojecting from {gdf.crs} to {target_crs}...")
                        gdf = gdf.to_crs(target_crs)
                    elif gdf.crs == target_crs:
                        print(f"  File is already in target CRS: {target_crs}.")
                else:
                    print(f"  No target_crs specified. Keeping original CRS: {gdf.crs}")

                # --- Save the processed GeoDataFrame ---
                # Construct output filename, preserving original name but changing extension if needed
                name_without_ext = os.path.splitext(filename)[0]
                output_filename = f"{name_without_ext}.{output_file_format}"
                output_file_path = os.path.join(output_folder_path, output_filename)

                # Remove kwargs that are only for reading, if any, before passing to to_file
                write_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["sep", "on_bad_lines", "encoding"]
                }

                # Specify driver for writing, especially for geojson/shapefile
                driver_map = {
                    "geojson": "GeoJSON",
                    "shp": "ESRI Shapefile",
                    "csv": "CSV",  # GeoPandas can write CSV, but geometry will be lost unless specified
                }
                driver = driver_map.get(output_file_format.lower(), None)
                if driver:
                    write_kwargs["driver"] = driver

                gdf.to_file(output_file_path, **write_kwargs)
                print(f"  Saved processed file to: {output_file_path}")
                processed_count += 1

            except Exception as e:
                print(f"  Error processing {filename}: {e}")
                print(f"  Skipping {filename}.")
                skipped_count += 1
        elif os.path.isdir(file_path):
            print(f"  Skipping directory: {filename}")
        else:
            print(f"  Skipping non-{file_extension} file: {filename}")

    print(
        f"\nStandardization and resaving complete. Processed {processed_count} files, skipped {skipped_count} files."
    )
