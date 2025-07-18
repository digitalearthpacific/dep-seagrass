
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