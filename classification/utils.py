
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
    data["evi"] = (2.5 * data["nir"] - data["red"]) / (data["nir"] + (6 * data["red"]) - (7.5 * data["blue"]) + 1)
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
    

def mask_deeps(
    ds: Dataset,
    ds_to_mask: Dataset | None = None,
    threshold: float = -0.02,
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
        collections=[collection],
        bbox=list(ds.odc.geobox.geographic_extent.boundingbox)
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
    _, land_mask = mask_land(ds, return_mask = True)
    _, deeps_mask = mask_deeps(ds, return_mask = True)
    _, elevation_mask = mask_elevation(ds, return_mask = True)
    
    mask = land_mask & deeps_mask & elevation_mask

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
        patch,
        distances=[1],
        angles=[0],
        levels=LEVELS,
        symmetric=True,
        normed=True
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
    output_dtype: str = 'float32',
    nodata_value: int = 255 # Value to use for NoData if output_dtype is an integer type
) -> xr.DataArray:
    """
"""

    # 1. Boolean mask for the target class
    target_class = (classification_da == target_class_id)

    # 2. Boolean mask for all non-NaN (classified) pixels
    is_classified = classification_da.notnull()

    # 3. Handle NaN.
    binary_output_float = xr.full_like(classification_da, np.nan, dtype=float)

    # 4. Set target class to 1.0 where target_class is True
    binary_output_float = xr.where(target_class, 1.0, binary_output_float)

    # 5. Set other classified areas to 0.0
    #    Other marine benethic habitats but not NaNs.
    binary_output_float = xr.where(is_classified & ~target_class, 0.0, binary_output_float)

    # 6. Handle final dtype conversion and NoData value
    if output_dtype in ['uint8', 'int8', 'int16', 'int32', 'int64']:
        # Check if the nodata_value is valid for the chosen dtype
        dtype_info = np.iinfo(output_dtype)
        if not (dtype_info.min <= nodata_value <= dtype_info.max):
            print(f"Warning: `nodata_value` ({nodata_value}) is outside the valid range "
                  f"for `output_dtype` ({output_dtype}) which is [{dtype_info.min}, {dtype_info.max}]. "
                  f"This might lead to unexpected results or data wrapping.")

        print(f"Casting to {output_dtype} and converting NaN (original no data) values to {nodata_value}.")
        # Replace NaNs with the specified nodata_value 
        final_output = binary_output_float.fillna(nodata_value).astype(output_dtype)
        # Add nodata attributes for GeoTIFF writing, crucial for GIS software
        final_output.attrs['_FillValue'] = nodata_value
        final_output.attrs['nodata'] = nodata_value # Common attribute for geospatial data
    else: # For float types, NaN is the natural nodata
        final_output = binary_output_float.astype(output_dtype)
        # Ensure _FillValue is set if there's an original nodata value in the input
        if '_FillValue' in classification_da.attrs:
            final_output.attrs['_FillValue'] = classification_da.attrs['_FillValue']
        elif 'nodata' in classification_da.attrs:
            final_output.attrs['nodata'] = classification_da.attrs['nodata']

    return final_output

def probability(
    ds: xr.Dataset,
    model, 
    bands: list[str],
    target_class_id: int, 
    nodata_value: int = 255, 
    # scale_to_100: bool = True # Whether to scale probabilities from 0-1 to 0-100
) -> xr.DataArray:
    """
    Makes an xarray.Dataset that includes the probability of seagrass
    """

    if not isinstance(ds, xr.Dataset):
        raise TypeError("Input 'ds' must be an xarray.Dataset.")
    if target_class_id not in model.classes_:
        raise ValueError(f"Target class ID {target_class_id} not found in model classes: {model.classes_}")

    print(f"Generating probability raster for class ID: {target_class_id}")

    # 1. Select the relevant bands and stack spatial dimensions
    features_ds = ds[bands]
    features_stacked = features_ds.stack(pixel=['y', 'x'])

    # 2. Convert to a Pandas DataFrame to easily handle NaNs
    features_df = features_stacked.to_dataframe()

    # Store the original index (pixel coordinates) before dropping NaNs/nodata_value
    original_pixel_index = features_df.index

    # Ensure all features are numeric before dropping NaNs
    features_df_numeric = features_df.apply(pd.to_numeric, errors='coerce')
    features_for_prediction_df = features_df_numeric.dropna()

    # Filter out rows where all feature values are equal to nodata_value
    if nodata_value is not None and not np.isnan(nodata_value):
        is_not_nodata_mask = (features_for_prediction_df != nodata_value).any(axis=1)
        features_for_prediction_df = features_for_prediction_df[is_not_nodata_mask]

    # Store the index of the valid (non-NaN, non-nodata) pixels that will be predicted
    valid_pixel_index = features_for_prediction_df.index

    # Convert the DataFrame to a NumPy array for sklearn
    features_for_prediction_np = features_for_prediction_df.values

    # 3. Get probabilities for all classes
    probabilities_np = model.predict_proba(features_for_prediction_np)

    # Find the index of the target class in the model's output
    target_class_index = list(model.classes_).index(target_class_id)

    # Extract probabilities for the target class
    target_class_probabilities_1d = probabilities_np[:, target_class_index]

    # 4. Create an empty DataArray with NaNs, matching the original spatial dimensions
    # This ensures that areas that were originally NaN (and thus dropped for prediction)
    # remain NaN in the output probability raster.
    # Use float dtype for probabilities
    probability_da = xr.full_like(ds[bands[0]], np.nan, dtype=float)
    probability_da.name = f'probability_class_{target_class_id}'

    # Fill the valid pixels with their predicted probabilities
    probability_da.loc[valid_pixel_index] = target_class_probabilities_1d

    # 5. Scale to 0-100 if requested
    if scale_to_100:
        probability_da = probability_da * 100

    # 6. Copy georeferencing attributes from a source band
    # This assumes the first band in `bands` list (`ds[bands[0]]`) has correct georeferencing
    probability_da.attrs = ds[bands[0]].attrs
    # Ensure CRS is maintained if using rioxarray
    # if hasattr(ds[bands[0]], 'rio'):
    #     probability_da = probability_da.rio.write_crs(ds[bands[0]].rio.crs)
    #     probability_da = probability_da.rio.write_transform(ds[bands[0]].rio.transform())

    print("Probability raster generation complete.")
    return probability_da