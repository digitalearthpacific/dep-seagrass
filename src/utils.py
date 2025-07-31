import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import xarray as xr


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


def texture(data: xr.DataArray, window_size=9, levels=32) -> xr.Dataset:
    # Input
    max = data.max().values
    min = data.min().values
    # Scale to 0-LEVELS for GLCM
    img = (
        ((data - min) / (max - min) * (levels - 1))
        .clip(0, levels - 1)
        .values.astype(np.uint8)
    )

    # Extract overlapping windows
    patches = sliding_window_view(img, (window_size, window_size))
    # Shape: (rows, cols, win_y, win_x)

    # Assuming 'patches' is a 4D NumPy array with dimensions (y_coords, x_coords, window_y_size, window_x_size)
    # To get the first patch (at y=0, x=0), you would index it like this:
    sample_patch_data = patches[0, 0, :, :]

    # Verify the shape of the extracted sample patch data
    print(f"Shape of sample_patch_data: {sample_patch_data.shape}")

    # Call glcm_features directly on this 2D sample data
    sample_result = glcm_features(sample_patch_data)

    # Use apply_ufunc to vectorize over (row, col) dimensions
    result = xr.apply_ufunc(
        glcm_features,
        xr.DataArray(patches, dims=["y", "x", "win_y", "win_x"]),
        input_core_dims=[["win_y", "win_x"]],
        output_core_dims=[["feature"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
    )

    # Add coordinates & names
    pad = window_size - 1
    result = result.assign_coords(
        {
            "y": data.y[:-pad],
            "x": data.x[:-pad],
            "feature": [
                "contrast",
                "homogeneity",
                "energy",
                "ASM",
                "correlation",
                "mean",
                "entropy",
            ],
        }
    )

    return result.to_dataset(dim="feature")


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
