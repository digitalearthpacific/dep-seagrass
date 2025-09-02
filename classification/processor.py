from logging import Logger

import xarray as xr
from dep_tools.processors import Processor
from masking import all_masks
from utils import (
    calculate_band_indices,
    do_prediction,
    extract_single_class,
    probability_binary,
    scale,
    texture,
)


class SeagrassProcessor(Processor):
    def __init__(
        self,
        model,
        probability_threshold: int = 60,
        nodata_value: int = 255,
        target_class_id: int = 4,
        fast_mode: bool = True,
        log: Logger = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model = model
        self._probability_threshold = probability_threshold
        self._nodata_value = nodata_value
        self._target_class_id = target_class_id
        self._fast_mode = fast_mode

        if log is None:
            self._log = Logger("Seagrass_processor")
        else:
            self._log = log

    def process(self, input_data: xr.Dataset) -> xr.Dataset:
        self._log.info("Starting processing of input data")
        # Scale data to values of 0-1 so that we can calculate indices properly
        scaled_data = scale(input_data).squeeze(drop=True)

        # Load data into memory here, before we do intensive things like texture
        self._log.info("Loading data into memory...")
        loaded_data = scaled_data.compute()

        # Compute indices
        self._log.info("Computing band indices...")
        data_indices = calculate_band_indices(loaded_data)

        # Calculate the texture data on unmasked data
        self._log.info("Computing texture data...")
        if self._fast_mode:
            self._log.info("Using fast texture calculation")
            texture_data = texture(data_indices.blue, fast=True, levels=32)
        else:
            self._log.info("Using standard texture calculation")
            texture_data = texture(data_indices.blue, fast=False, levels=32)

        texture_data = texture_data.compute()

        # Combine the two datasets before applying the mask
        combined_data = xr.merge([data_indices, texture_data])

        # Mask all the data
        self._log.info("Applying masks...")
        masked_scaled = all_masks(combined_data, return_mask=False, log=self._log)

        # Run the prediction
        self._log.info("Running prediction and probability process...")
        classification, probability = do_prediction(
            masked_scaled, self._model, self._target_class_id
        )

        seagrass_threshold = probability_binary(
            probability,
            self._probability_threshold,
            nodata_value=self._nodata_value,
        )

        seagrass_class = extract_single_class(
            classification,
            self._target_class_id,
        )

        output = xr.Dataset(
            {
                "classification": classification,
                "seagrass_probability": probability,
                "seagrass_threshold_60": seagrass_threshold,
                "seagrass": seagrass_class,
            }
        )

        for var in output.data_vars:
            output[var].odc.nodata = self._nodata_value
            output[var].attrs["_FillValue"] = self._nodata_value

        return output
