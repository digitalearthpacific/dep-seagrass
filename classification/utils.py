# utils.py

# Standard library imports (alphabetical within category is good practice)
from datetime import datetime

# Third-party library imports
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import pyproj # Imported pyproj to get CRS
import rasterio as rio
import rioxarray
import xarray as xr
from ipyleaflet import basemaps
from numpy.lib.stride_tricks import sliding_window_view
from odc.stac import load
from pystac.client import Client
from shapely import box
from shapely.geometry import Polygon
from skimage.feature import graycomatrix, graycoprops

# --- Re-export commonly used objects/modules ---
# This list explicitly defines what names will be available when you do
# 'from utils import ...'. It's good practice for clarity.
# Note: For `pyproj.CRS`, we import `pyproj` then re-export `CRS` directly.
__all__ = [
    # Re-exported modules/objects
    "datetime",
    "Polygon",
    "box",
    "CRS", # Re-exporting CRS directly from pyproj
    "folium",
    "gpd",
    "np",
    "pd",
    "rio",
    "xr",
    "rioxarray",
    "basemaps",
    "sliding_window_view",
    "pystac_client",
    "planetary_computer",
    "load", # odc.stac.load
    "Client", # pystac.client.Client
    "graycomatrix",
    "graycoprops",

    # Your custom utility functions (assuming their definitions are below)
    "scale",
    "do_prediction",
    "calculate_band_indices",
    "apply_masks",
    "threshold_calc_land",
    "threshold_calc_ds"
]

# Explicitly import CRS from pyproj to make it available for re-export
from pyproj import CRS

# --- Your Custom Utility Functions ---
# Paste the actual definitions of your functions here.
# These are just placeholders:



from odc.stac import load  # Correct source for `load`
import xarray as xr
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import data
from skimage.util import view_as_windows

def load_data(items, bbox):
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
        measurements=[
            "nir",
            "red",
            "blue",
            "green",
            "emad",
            "smad",
            "bcmad",
            "count",
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
    data["ndwi"] = (data["green"] - data["nir"]) / (data["green"] + data["nir"])
    data["b_g"] = data["blue"] / data["green"]
    data["b_r"] = data["blue"] / data["red"]
    data["mci"] = data["nir"] / data["rededge1"]
    data["ndci"] = (data["rededge1"] - data["red"]) / (data["rededge1"] + data["red"])
    # additional indices from SDB (Alex)*
    # data['stumpf'] = np.log(np.abs(data.green - data.blue)) / np.log(data.green + data.blue)
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


def apply_masks(data):
    """
    Apply a series of masks to the dataset based on specific indices.

    Parameters:
    data (xr.Dataset): The input dataset containing the necessary bands.

    Returns:
    xr.Dataset: The dataset after applying the masks.
    """
    mndwi = (data["green"] - data["swir16"]) / (data["green"] + data["swir16"])
    # Major land mask
    # mndwi_land_mask = mndwi > 0
    # Moderate land mask
    # mdnwi_land_mask = mndwi > -0.35
    # Minor land mask
    mndwi_land_mask = mndwi > -0.5
    masked_data = data.where(mndwi_land_mask)
    ndti = (masked_data["red"] - masked_data["green"]) / (
        masked_data["red"] + masked_data["green"]
    )
    ndti_mask = ndti < 0.2
    masked_data = masked_data.where(ndti_mask)
    # Major NIR mask 
    # nir_mask = masked_data["nir"] < 0.085
    # Conservative NIR mask
    nir_mask = masked_data["nir"] < 0.8
    masked_data = masked_data.where(nir_mask)

    return masked_data

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
    predicted_da = predicted_da.where(~mask).compute()

    # If we have a name, return dataset, else the dataarray
    if output_name is None:
        return predicted_da
    else:
        return predicted_da.to_dataset(name=output_name)

def glcm_features(patch):
    # --- Handle RuntimeWarning (keep these checks as they are good practice) ---
    # This addresses cases where patches are uniform or contain NaNs.
    if np.all(np.isnan(patch)) or (np.nanmax(patch) == np.nanmin(patch) and not np.isnan(np.nanmax(patch))):
        # Return default values for the 4 features you want: contrast, homogeneity, energy, correlation
        return np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)

    try:
        glcm_input = patch.astype(np.uint8)
    except ValueError:
        # If conversion to uint8 fails (e.g., due to remaining NaNs/infs), return defaults
        print("Warning: Failed to convert patch to uint8 for GLCM. Returning defaults.")
        return np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)

    # --- GLCM Calculation ---
    glcm = graycomatrix(glcm_input, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=np.max(glcm_input) + 1 if glcm_input.size > 0 else 256,
                        symmetric=True, normed=True)

    # --- Initialize 'out' for the DESIRED features ---
    # You want 'contrast', 'homogeneity', 'energy', 'correlation'. That's 4 features.
    out = np.zeros(5, dtype=np.float32) # <--- THIS MUST BE 4!

    # --- Assign GLCM properties ---
    out[0] = graycoprops(glcm, "contrast")[0, 0]
    out[1] = graycoprops(glcm, "homogeneity")[0, 0]
    out[2] = graycoprops(glcm, "energy")[0, 0]
    out[3] = graycoprops(glcm, "ASM")[0, 0]
    out[4] = graycoprops(glcm, "correlation")[0, 0]
    out[5] = graycoprops(glcm, "mean")[0, 0]

    return out

def threshold_calc_land(band, level=None):
    """
    Calculates threshold for a band.

    Parameters:
    band (xr.DataArray): The input band as an xarray DataArray.
    level (str, optional): Threshold level, must be one of 'High', 'Mid', or 'Low'. Defaults to 'Low' if not provided.

    Returns:
    float: The calculated threshold value for the specified level.
    """

    # mean 
    mean = band.mean().compute().item() 
    # standard deviation
    std = band.std().compute().item()
    
    thresh_moderate = mean
    thresh_minor = mean - std
    thresh_major = mean + std

    # Default to "Low" if level is None or not provided
    if level is None or level == "Low":
        return round(thresh_minor, 3)
    elif level == "High":
        return round(thresh_major, 3)
    elif level == "Mid":
        return round(thresh_moderate, 3)
    else:
        raise ValueError("level must be one of: 'High', 'Mid', or 'Low'")


def threshold_calc_ds(band, level=None):
    """
    Calculates threshold for a band.

    Parameters:
    band (xr.DataArray): The input band as an xarray DataArray.
    level (str, optional): Threshold level, must be one of 'High', 'Mid', or 'Low'. Defaults to 'Low' if not provided.

    Returns:
    float: The calculated threshold value for the specified level.
    """

    # mean 
    mean = band.mean().compute().item() 
    # standard deviation
    std = band.std().compute().item()
    
    thresh_moderate = mean
    thresh_minor = mean + std
    thresh_major = mean - std

    # Default to "Low" if level is None or not provided
    if level is None or level == "Low":
        return round(thresh_minor, 3)
    elif level == "High":
        return round(thresh_major, 3)
    elif level == "Mid":
        return round(thresh_moderate, 3)
    else:
        raise ValueError("level must be one of: 'High', 'Mid', or 'Low'")

    