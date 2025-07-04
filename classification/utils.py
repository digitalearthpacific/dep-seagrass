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


def patchwise_glcm_feature(
    image_blue, window_size=9, levels=256
):

    pad = window_size // 2
    padded = np.pad(image_blue, pad, mode='reflect')
    windows = view_as_windows(padded, (window_size, window_size))
    H, W = windows.shape[:2]

    # Pre-allocate feature arrays
    contrast = np.zeros((H, W), dtype=np.float32)
    homogeneity = np.zeros((H, W), dtype=np.float32)
    entropy = np.zeros((H, W), dtype=np.float32)
    energy = np.zeros((H, W), dtype=np.float32)
    correlation = np.zeros((H, W), dtype=np.float32)
    mean = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            patch = windows[i, j].astype('uint8')  # Ensure patch is uint8
            glcm = graycomatrix(
                patch,
                distances=[1],
                angles=[0],
                levels=levels,
                symmetric=True,
                normed=True
            )
            contrast[i, j] = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity[i, j] = graycoprops(glcm, 'homogeneity')[0, 0]
            energy[i, j] = graycoprops(glcm, 'energy')[0, 0]
            mean[i, j] = graycoprops(glcm, 'mean')[0, 0]
            correlation[i, j] = graycoprops(glcm, 'correlation')[0, 0]
            glcm_p = glcm[:, :, 0, 0]
            entropy[i, j] = -np.sum(glcm_p * np.log2(glcm_p + 1e-10))

    return {
        'contrast': contrast,
        'homogeneity': homogeneity,
        'entropy': entropy,
        'energy': energy,
        'correlation': correlation,
        'mean': mean
    }