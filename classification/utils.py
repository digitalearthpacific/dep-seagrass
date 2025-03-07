from dask.distributed import Client as DaskClient
from odc.stac import load, configure_s3_access  # Correct source for `load`


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
            "nir", "red", "blue", "green", "emad", "smad", 
            "bcmad", "count", "green", "nir08", 
            "nir09", "swir16", "swir22", "coastal",
            "rededge1", "rededge2", "rededge3", 
        ],
        bbox=bbox,
        chunks={"x": 2048, "y": 2048},
        groupby="solar_day",
    )
    return data

import xarray as xr


def assign_band_coords(dataset, band_names):
    """
    Assigns band names as coordinates in an xarray.Dataset while retaining all data variables.

    Parameters:
    -----------
    dataset : xarray.Dataset
        An xarray Dataset where multiple bands exist as separate data variables.
    band_names : list
        A list of band names to assign as coordinates.

    Returns:
    --------
    xarray.Dataset
        The same Dataset but with a 'band' coordinate assigned.
    """

    # Ensure input is an xarray.Dataset
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("Input must be an xarray.Dataset")

    # Convert Dataset to DataArray (stack all bands)
    stacked = dataset.to_array(dim="band")

    # Ensure number of band names matches
    if len(band_names) != stacked.sizes["band"]:
        raise ValueError(f"Number of band names ({len(band_names)}) does not match the number of bands ({stacked.sizes['band']}).")

    # Assign band names as coordinates
    stacked = stacked.assign_coords({"band": band_names})

    # Convert back to Dataset (retain data variables)
    dataset = stacked.to_dataset(dim="band")

    return dataset


    
band_names = [
    "nir", "red", "blue", "green", "emad", "smad", "bcmad", "count",
    "nir08", "nir09", "swir16", "swir22", "coastal",
    "rededge1", "rededge2", "rededge3"
]

def calculate_band_indices(data):
    """
    Calculate various band indices and add them to the dataset.

    Parameters:
    data (xarray.Dataset): The input dataset containing the necessary spectral bands.

    Returns:
    xarray.Dataset: The dataset with added band indices.
    """

    data['mndwi'] = (data['green'] - data['swir16']) / (data['green'] + data['swir16'])
    data['ndti'] = (data['red'] - data['green']) / (data['red'] + data['green'])
    data['cai'] = (data['coastal'] - data['blue']) / (data['coastal'] + data['blue'])
    data['ndvi'] = (data['nir'] - data['red']) / (data['nir'] + data['red'])
    data['ndwi'] = (data['green'] - data['nir']) / (data['green'] + data['nir'])
    data['b_g'] = data['blue'] / data['green']
    data['b_r'] = data['blue'] / data['red']
    data['mci'] = data['nir'] / data['rededge1']
    data['ndci'] = (data['rededge1'] - data['red']) / (data['rededge1'] + data['red'])

    return data


def add_spectral_indices(ds):
    """
    Compute and add spectral indices to the input xarray.Dataset.

    """

    indices = {
        'mndwi': (ds['green'] - ds['swir16']) / (ds['green'] + ds['swir16']),
        'ndti': (ds['red'] - ds['green']) / (ds['red'] + ds['green']),
        'cai': (ds['coastal'] - ds['blue']) / (ds['coastal'] + ds['blue']),
        'ndvi': (ds['nir'] - ds['red']) / (ds['nir'] + ds['red']),
        'ndwi': (ds['green'] - ds['nir']) / (ds['green'] + ds['nir']),
        'b_g': ds['blue'] / ds['green'],
        'b_r': ds['blue'] / ds['red'],
        'mci': ds['nir'] / ds['rededge1'],
        'ndci': (ds['rededge1'] - ds['red']) / (ds['rededge1'] + ds['red'])
    }

    for index_name, index_values in indices.items():
        ds[index_name] = index_values

    return ds

# clipped_ds = data
# nir = clipped_ds["nir"]
# mndwi = (clipped_ds["green"] - clipped_ds["swir16"]) / (clipped_ds["green"] + clipped_ds["swir16"])
# ndti = (clipped_ds["red"] - clipped_ds["green"]) / (clipped_ds["red"] + clipped_ds["green"])
# cai = (clipped_ds["coastal"]-clipped_ds["blue"])/( clipped_ds["coastal"]+ clipped_ds["blue"]) #coastal aerosol index
# ndvi = (clipped_ds["nir"]-clipped_ds["red"])/( clipped_ds["nir"]+ clipped_ds["red"]) #vegetation index (NDVI)
# ndwi = (clipped_ds["green"]-clipped_ds["nir"])/(clipped_ds["green"]+clipped_ds["nir"]) #water index (NDWI)
# b_g = (clipped_ds["blue"])/(clipped_ds["green"]) #blue to green ratio
# b_r = (clipped_ds["blue"])/(clipped_ds["red"]) #blue to red ratio
# mci = (clipped_ds["nir"])/(clipped_ds["rededge1"]) # max chlorophlyll index (MCI)
# ndci = (clipped_ds["rededge1"]-clipped_ds["red"])/(clipped_ds["rededge1"]+clipped_ds["red"]) # normalised difference chlorophyll index (NDCI)


def scale(data):
    """
    Scale the input data by applying a factor and clipping the values.

    Parameters:
    data (xr.Dataset): The input dataset containing the bands to be scaled.

    Returns:
    xr.Dataset: The scaled dataset with values clipped between 0 and 1.
    """
    scaled = (data.where(data != 0) * 0.0001).clip(0, 1)
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
    mndwi_land_mask = mndwi > 0
    masked_data = data.where(mndwi_land_mask)
    ndti = (masked_data["red"] - masked_data["green"]) / (masked_data["red"] + masked_data["green"])
    ndti_mask = ndti < 0.2
    masked_data = masked_data.where(ndti_mask)
    nir_mask = masked_data['nir'] < 0.085
    masked_data = masked_data.where(nir_mask)

    return masked_data

from dask.distributed import Client as DaskClient
from odc.stac import load, configure_s3_access  # Correct source for `load`

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
            "nir", "red", "blue", "green", "emad", "smad", 
            "bcmad", "count", "green", "nir08", 
            "nir09", "swir16", "swir22", "coastal",
            "rededge1", "rededge2", "rededge3", 
        ],
        bbox=bbox,
        chunks={"x": 2048, "y": 2048},
        groupby="solar_day",
    )
    return data

import xarray as xr


def assign_band_coords(dataset, band_names):
    """
    Assigns band names as coordinates in an xarray.Dataset while retaining all data variables.

    Parameters:
    -----------
    dataset : xarray.Dataset
        An xarray Dataset where multiple bands exist as separate data variables.
    band_names : list
        A list of band names to assign as coordinates.

    Returns:
    --------
    xarray.Dataset
        The same Dataset but with a 'band' coordinate assigned.
    """

    # Ensure input is an xarray.Dataset
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("Input must be an xarray.Dataset")

    # Convert Dataset to DataArray (stack all bands)
    stacked = dataset.to_array(dim="band")

    # Ensure number of band names matches
    if len(band_names) != stacked.sizes["band"]:
        raise ValueError(f"Number of band names ({len(band_names)}) does not match the number of bands ({stacked.sizes['band']}).")

    # Assign band names as coordinates
    stacked = stacked.assign_coords({"band": band_names})

    # Convert back to Dataset (retain data variables)
    dataset = stacked.to_dataset(dim="band")

    return dataset


    
band_names = [
    "nir", "red", "blue", "green", "emad", "smad", "bcmad", "count",
    "nir08", "nir09", "swir16", "swir22", "coastal",
    "rededge1", "rededge2", "rededge3"
]

def calculate_band_indices(data):
    """
    Calculate various band indices and add them to the dataset.

    Parameters:
    data (xarray.Dataset): The input dataset containing the necessary spectral bands.

    Returns:
    xarray.Dataset: The dataset with added band indices.
    """

    data['mndwi'] = (data['green'] - data['swir16']) / (data['green'] + data['swir16'])
    data['ndti'] = (data['red'] - data['green']) / (data['red'] + data['green'])
    data['cai'] = (data['coastal'] - data['blue']) / (data['coastal'] + data['blue'])
    data['ndvi'] = (data['nir'] - data['red']) / (data['nir'] + data['red'])
    data['ndwi'] = (data['green'] - data['nir']) / (data['green'] + data['nir'])
    data['b_g'] = data['blue'] / data['green']
    data['b_r'] = data['blue'] / data['red']
    data['mci'] = data['nir'] / data['rededge1']
    data['ndci'] = (data['rededge1'] - data['red']) / (data['rededge1'] + data['red'])

    return data


def add_spectral_indices(ds):
    """
    Compute and add spectral indices to the input xarray.Dataset.

    """

    indices = {
        'mndwi': (ds['green'] - ds['swir16']) / (ds['green'] + ds['swir16']),
        'ndti': (ds['red'] - ds['green']) / (ds['red'] + ds['green']),
        'cai': (ds['coastal'] - ds['blue']) / (ds['coastal'] + ds['blue']),
        'ndvi': (ds['nir'] - ds['red']) / (ds['nir'] + ds['red']),
        'ndwi': (ds['green'] - ds['nir']) / (ds['green'] + ds['nir']),
        'b_g': ds['blue'] / ds['green'],
        'b_r': ds['blue'] / ds['red'],
        'mci': ds['nir'] / ds['rededge1'],
        'ndci': (ds['rededge1'] - ds['red']) / (ds['rededge1'] + ds['red'])
    }

    for index_name, index_values in indices.items():
        ds[index_name] = index_values

    return ds

# clipped_ds = data
# nir = clipped_ds["nir"]
# mndwi = (clipped_ds["green"] - clipped_ds["swir16"]) / (clipped_ds["green"] + clipped_ds["swir16"])
# ndti = (clipped_ds["red"] - clipped_ds["green"]) / (clipped_ds["red"] + clipped_ds["green"])
# cai = (clipped_ds["coastal"]-clipped_ds["blue"])/( clipped_ds["coastal"]+ clipped_ds["blue"]) #coastal aerosol index
# ndvi = (clipped_ds["nir"]-clipped_ds["red"])/( clipped_ds["nir"]+ clipped_ds["red"]) #vegetation index (NDVI)
# ndwi = (clipped_ds["green"]-clipped_ds["nir"])/(clipped_ds["green"]+clipped_ds["nir"]) #water index (NDWI)
# b_g = (clipped_ds["blue"])/(clipped_ds["green"]) #blue to green ratio
# b_r = (clipped_ds["blue"])/(clipped_ds["red"]) #blue to red ratio
# mci = (clipped_ds["nir"])/(clipped_ds["rededge1"]) # max chlorophlyll index (MCI)
# ndci = (clipped_ds["rededge1"]-clipped_ds["red"])/(clipped_ds["rededge1"]+clipped_ds["red"]) # normalised difference chlorophyll index (NDCI)


def scale(data):
    """
    Scale the input data by applying a factor and clipping the values.

    Parameters:
    data (xr.Dataset): The input dataset containing the bands to be scaled.

    Returns:
    xr.Dataset: The scaled dataset with values clipped between 0 and 1.
    """
    scaled = (data.where(data != 0) * 0.0001).clip(0, 1)
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
    mndwi_land_mask = mndwi > 0
    masked_data = data.where(mndwi_land_mask)
    ndti = (masked_data["red"] - masked_data["green"]) / (masked_data["red"] + masked_data["green"])
    ndti_mask = ndti < 0.2
    masked_data = masked_data.where(ndti_mask)
    nir_mask = masked_data['nir'] < 0.085
    masked_data = masked_data.where(nir_mask)

    return masked_data