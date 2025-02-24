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

def mask_and_scale(data):
    """
    Applies a Sentinel-2 cloud mask and scales the data values.

    Parameters:
    - data (xarray.DataArray): The input data array containing Sentinel-2 data with a 'scl' band (scene classification layer).

    Returns:
    - xarray.DataArray: The masked and scaled data.
    """
    # Mask out clouds and scale values
    # Sentinel-2 cloud mask flags (1: defective, 3: shadow, 9: high confidence cloud, 10: thin cirrus)
    mask_flags = [1, 3, 9, 10]

    # Apply the cloud mask (invert the mask to keep non-cloud values)
    cloud_mask = ~data.scl.isin(mask_flags)
    masked = data.where(cloud_mask)

    # Apply scaling and clip values from 0 to 1
    scaled = (masked.where(masked != 0) * 0.0001).clip(0, 1)
    
    return scaled

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


# def assign_band_coords(dataset, band_names):
     
#     """
#     Converts an xarray.Dataset with multiple bands into an xarray.DataArray 
#     with a 'band' dimension and assigns band names as coordinates.

#     Parameters:
#     -----------
#     dataset : xarray.Dataset
#         An xarray Dataset where each band is a separate data variable.
#     band_names : list
#         A list of band names to assign as coordinates.

#     Returns:
#     --------
#     xarray.DataArray
#         A DataArray with a 'band' dimension and assigned band names.
#     """
    
#     # Ensure input is a Dataset
#     if not isinstance(dataset, xr.Dataset):
#         raise TypeError("Input must be an xarray.Dataset")

#     # Convert Dataset to DataArray
#     dataset = dataset.to_array(dim="band")

#     # Check if number of band names matches
#     if len(band_names) != dataset.sizes["band"]:
#         raise ValueError(f"Number of band names ({len(band_names)}) does not match the number of bands ({data_array.sizes['band']}).")

#     # Assign band names as coordinates
#     dataset = dataset.assign_coords({"band": band_names})

#     # Ensure dimensions are in the correct order (band, y, x)
#     expected_dims = ("band", "time", "y", "x")
#     dataset = dataset.transpose(*[dim for dim in expected_dims if dim in dataset.dims])

#     return dataset
    
band_names = [
    "nir", "red", "blue", "green", "emad", "smad", "bcmad", "count",
    "nir08", "nir09", "swir16", "swir22", "coastal",
    "rededge1", "rededge2", "rededge3"
]