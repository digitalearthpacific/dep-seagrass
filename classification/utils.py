import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from numba import njit
from skimage.feature import graycomatrix, graycoprops


def scale(data):
    """
    Scale the input data by applying a factor and clipping the values.

    Parameters:
    data (xr.Dataset): The input dataset containing the bands to be scaled.

    Returns:
    xr.Dataset: The scaled dataset with values clipped between 0 and 1.
    """
    mad_bands = ["smad", "emad", "bcmad"]
    data_mad = data[mad_bands]
    non_mad_bands = data[list(set(data.data_vars) - set(mad_bands))].astype("float32")
    scaled = (non_mad_bands * 0.0001).clip(0, 1)

    return xr.merge([scaled, data_mad])


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


def texture(
    data: xr.DataArray,
    window_size: int = 7,
    levels: int = 64,
    chunk_size: int = 600,
    fast: bool = True,
) -> xr.Dataset:
    min_val = data.min()
    max_val = data.max()
    img = (((data - min_val) / (max_val - min_val)) * (levels - 1)).astype(np.float32)

    # Force Dask backing
    if not isinstance(img.data, da.Array):
        img = xr.DataArray(
            da.from_array(img.values),
            dims=img.dims,
            coords=img.coords,
            name=img.name,
        )
    img = img.chunk({"x": chunk_size, "y": chunk_size})

    func = glcm_features_fast if fast else glcm_features

    patches_xarr = img.rolling(x=window_size, y=window_size, center=False).construct(
        y="win_y", x="win_x"
    )

    # Run map_overlap
    result = xr.apply_ufunc(
        func,
        patches_xarr,
        input_core_dims=[["win_y", "win_x"]],
        output_core_dims=[["feature"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={"output_sizes": {"feature": 7}},
        kwargs={"levels": levels},
    )

    # Wrap result with original coords
    texture_ds = xr.DataArray(
        result,
        dims=("y", "x", "feature"),
        coords={
            "y": data.y,
            "x": data.x,
            "feature": [
                "contrast",
                "homogeneity",
                "energy",
                "ASM",
                "correlation",
                "mean",
                "entropy",
            ],
        },
    ).to_dataset(dim="feature")

    return texture_ds


@njit(cache=True, fastmath=True)
def glcm_features_fast(patch: np.ndarray, levels: int) -> np.ndarray:
    """
    Chat GPT-authored code to do the processing faster.

    Prompt-engineered by Alex, who doesn't know if it actually works...
    """
    # All-NaN window → all-NaN features
    if np.isnan(patch).all():
        return np.empty(7, dtype=np.float32)

    # Build symmetric GLCM for distance=1, angle=0 (horizontal),
    # skipping any pair where either value is NaN
    glcm = np.zeros((levels, levels), dtype=np.float64)
    h, w = patch.shape

    # also track mean over valid pixels (exclude NaNs)
    sum_pix = 0.0
    count_pix = 0

    # pre-clipped & quantized indices
    for i in range(h):
        for j in range(w):
            v = patch[i, j]
            if not np.isnan(v):
                sum_pix += v
                count_pix += 1

    for i in range(h):
        for j in range(w - 1):
            a = patch[i, j]
            b = patch[i, j + 1]
            if np.isnan(a) or np.isnan(b):
                continue
            ia = int(a + 0.5)  # round to nearest bin
            ib = int(b + 0.5)
            if ia < 0:
                ia = 0
            elif ia >= levels:
                ia = levels - 1
            if ib < 0:
                ib = 0
            elif ib >= levels:
                ib = levels - 1
            glcm[ia, ib] += 1.0
            glcm[ib, ia] += 1.0  # symmetric

    s = glcm.sum()
    out = np.empty(7, dtype=np.float32)
    if s == 0.0:
        # No valid neighbor pairs → features undefined
        out[:] = np.nan
        return out

    P = glcm / s

    # means and stds of marginal distributions
    mu_i = 0.0
    mu_j = 0.0
    for i in range(levels):
        for j in range(levels):
            pij = P[i, j]
            mu_i += i * pij
            mu_j += j * pij

    sigma_i_acc = 0.0
    sigma_j_acc = 0.0
    for i in range(levels):
        di = i - mu_i
        for j in range(levels):
            pij = P[i, j]
            dj = j - mu_j
            sigma_i_acc += di * di * pij
            sigma_j_acc += dj * dj * pij
    sigma_i = np.sqrt(sigma_i_acc)
    sigma_j = np.sqrt(sigma_j_acc)

    # features
    contrast = 0.0
    homogeneity = 0.0
    asm = 0.0
    entropy = 0.0
    corr_num = 0.0

    for i in range(levels):
        for j in range(levels):
            pij = P[i, j]
            if pij <= 0.0:
                continue
            diff = i - j
            contrast += diff * diff * pij
            homogeneity += pij / (1.0 + abs(diff))
            asm += pij * pij
            entropy -= pij * np.log2(pij)
            corr_num += (i - mu_i) * (j - mu_j) * pij

    energy = np.sqrt(asm)
    if sigma_i > 0.0 and sigma_j > 0.0:
        correlation = corr_num / (sigma_i * sigma_j)
    else:
        correlation = np.nan

    # mean intensity of the patch (exclude NaNs)
    mean_val = (sum_pix / count_pix) if count_pix > 0 else np.nan

    out[0] = np.float32(contrast)
    out[1] = np.float32(homogeneity)
    out[2] = np.float32(energy)
    out[3] = np.float32(asm)
    out[4] = np.float32(correlation)
    out[5] = np.float32(mean_val)
    out[6] = np.float32(entropy)

    return out


def glcm_features(patch: np.ndarray, levels: int) -> np.ndarray:
    # If there is no data, return nan for all vars
    # AGL: I have no idea if it works
    if np.isnan(patch).all():
        return np.full(7, float("nan"))

    # Careful here as it casts any nans to 0 (not sure what is desired!)
    patch = np.array(patch, copy=True).astype("uint8")
    glcm = graycomatrix(
        patch, distances=[1], angles=[0], levels=levels, symmetric=True, normed=True
    )
    out = np.empty(7, dtype=np.float32)
    out[0] = graycoprops(glcm, "contrast")[0, 0]
    out[1] = graycoprops(glcm, "homogeneity")[0, 0]
    out[2] = graycoprops(glcm, "energy")[0, 0]
    out[3] = graycoprops(glcm, "ASM")[0, 0]
    out[4] = graycoprops(glcm, "correlation")[0, 0]
    out[5] = graycoprops(glcm, "mean")[0, 0]

    glcm_p = glcm[:, :, 0, 0]
    out[6] = -np.sum(glcm_p * np.log2(glcm_p + 1e-10))

    return out


def reshape_array_to_2d(
    stacked_array: np.ndarray, template_ds: xr.DataArray, original_mask: xr.DataArray
) -> xr.Dataset:
    """Reshapes a 1D array into a 2D Dataset."""
    # Reshape back to the original 2D array
    array = stacked_array.to_numpy().reshape(template_ds.y.size, template_ds.x.size)

    # Convert to an xarray
    da = xr.DataArray(
        array, coords={"y": template_ds.y, "x": template_ds.x}, dims=["y", "x"]
    )

    # Return the masked dataarray, with 255 for nodata areas
    return da.where(~original_mask, 255).astype("uint8")


def do_prediction(ds, model, target_class_id: int = 4):
    """Predicts the model on the dataset and adds the prediction as a new variable.

    Args:
        ds (Dataset): Dataset to predict on
        model (RegressorMixin): Model to predict with
        target_class_id (int): ID of the target class for prediction

    Returns:
        Dataset: Dataset with the prediction as a new variable
    """
    # Store the original mask
    mask = ds.red.isnull()  # Probably should check more bands

    # Convert to a stacked array of observations
    stacked_arrays = ds.to_array().stack(dims=["y", "x"])

    # Replace any infinities with NaN
    stacked_arrays = stacked_arrays.where(stacked_arrays != float("inf"))
    stacked_arrays = stacked_arrays.where(stacked_arrays != float("-inf"))

    # Replace any NaN values with 0 and transpose to the right shape
    stacked_arrays = stacked_arrays.squeeze().fillna(0).transpose().to_pandas()

    # Remove the all-zero rows
    # This should make it MUCH MUCH faster, as we're not processing masked areas
    zero_mask = (stacked_arrays == 0).all(axis=1)
    non_zero_df = stacked_arrays.loc[~zero_mask]

    # Create a new array to hold the predictions
    full_predictions = pd.Series(np.nan, index=stacked_arrays.index)
    full_probabilities = pd.Series(np.nan, index=stacked_arrays.index)

    # Only run the prediction if there are non-zero rows
    if not non_zero_df.empty:
        # Predict the classes
        predictions = model.predict(non_zero_df)
        full_predictions.loc[~zero_mask] = predictions

        # Do the same for the probabilities
        probabilities = model.predict_proba(non_zero_df)
        target_class_index = list(model.classes_).index(target_class_id)
        target_probabilities = probabilities[:, target_class_index]
        target_probabilities_scaled = target_probabilities * 100
        full_probabilities.loc[~zero_mask] = target_probabilities_scaled

    # Reshape the results
    predicted = reshape_array_to_2d(full_predictions, ds, mask)
    probabilities = reshape_array_to_2d(full_probabilities, ds, mask)

    # Results should both be uint8 with 255 as nodata
    return predicted, probabilities


def probability_binary(
    probability_da: xr.DataArray,
    threshold: int | float,
    nodata_value: int = 255,
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
    mask = probability_da == 255
    above_threshold = probability_da >= threshold

    final_output = xr.where(above_threshold, 1, 0)
    final_output = xr.where(mask, nodata_value, final_output).astype("uint8")

    return final_output


def extract_single_class(
    classification: xr.DataArray, target_class_id: int, nodata_value: int = 255
) -> xr.DataArray:
    one_class = classification == target_class_id
    one_class = one_class.where(one_class == 1, 0)
    one_class = one_class.where(~(classification == nodata_value), nodata_value).astype(
        "uint8"
    )

    return one_class
