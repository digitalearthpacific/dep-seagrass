
from odc.stac import load  # Correct source for `load`
import xarray as xr
import numpy as np
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

def elevation_mask(
    asset_href: str,
    bbox: tuple,
    elevation_threshold: float = 10.0,
    bbox_crs: str = "EPSG:4326", # CRS of the input bbox tuple
    target_dem_crs: str = "EPSG:3832" # Target CRS for DEM and clipping polygon
) -> xr.DataArray:
    """
    Masks a Digital Elevation Model (DEM) based on an elevation threshold
    within a specified bounding box.

    Args: 
    1. asset_href (str): URL or local path to the DEM raster asset.
    2. bbox (tuple): Bounding box as (min_x, min_y, max_x, max_y). This tuple is assumed to be in `bbox_crs`.
    3. elevation_threshold (float): Elevation value (in the DEM's units, typically meters) below which to retain data.
    4. bbox_crs (str): The CRS (e.g., "EPSG:4326") of the input `bbox` tuple. Defaults to "EPSG:4326" (longitude/latitude). Advise to be reprojected to 3832 afterwards, 
    5. target_dem_crs (str): The CRS (e.g., "EPSG:3832") to which the DEM and
                              
    Returns:
        xarray.DataArray: The masked DEM data, where areas above the
                          elevation threshold are set to NaN. The data will be
                          in `target_dem_crs`.
    """
    # 1. Create shapely polygon from the input bbox
    # This polygon is initially in `bbox_crs`
    bbox_polygon_original_crs = box(*bbox)

    # 2. Load the DEM asset
    dem = rioxarray.open_rasterio(asset_href).squeeze()

    # 3. Ensure DEM has a CRS and reproject it to the target_dem_crs
    if dem.rio.crs is None:
        print(f"Warning: DEM has no CRS specified. Assuming {target_dem_crs} for initial operations.")
        # Attempt to assign CRS if missing; if it's incorrect, subsequent reprojection might fail.
        # It's better if the source DEM inherently has its CRS defined.
        dem = dem.rio.write_crs(target_dem_crs)
    elif str(dem.rio.crs) != target_dem_crs:
        print(f"Reprojecting DEM from {dem.rio.crs} to {target_dem_crs} for consistency...")
        dem = dem.rio.reproject(target_dem_crs)

    # 4. Reproject the clipping polygon to match the target_dem_crs
    # This is crucial to ensure the clipping geometry is in the same CRS as the DEM
    bbox_polygon_for_clip = bbox_polygon_original_crs
    if bbox_crs != target_dem_crs:
        print(f"Reprojecting clipping polygon from {bbox_crs} to {target_dem_crs}...")
        # Use geopandas to reproject the shapely polygon
        gdf = gpd.GeoSeries([bbox_polygon_original_crs], crs=bbox_crs)
        bbox_polygon_for_clip = gdf.to_crs(target_dem_crs).iloc[0]

    # 5. Clip the DEM using the reprojected polygon
    # The `crs` argument here specifies the CRS of the `geometries` being passed.
    dem_clipped = dem.rio.clip([bbox_polygon_for_clip], crs=target_dem_crs, drop=True)

    # 6. Apply the elevation mask
    # The original code `masked = dem.where(dem <= elevation_threshold)`
    # means "keep values that are less than or equal to the threshold,
    # and set others (greater than threshold) to NaN".
    # We apply this to the clipped DEM.
    masked_dem = dem_clipped.where(dem_clipped <= elevation_threshold)

    # 7. Compute the result to bring it into memory (if it's a Dask array)
    # This ensures the result is ready for immediate use.
    masked_dem = masked_dem.compute()

    return masked_dem



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

    