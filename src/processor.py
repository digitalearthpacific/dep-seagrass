from dep_tools.processors import Processor
import xarray as xr

from masking import all_masks
from utils import (
    calculate_band_indices,
    scale,
    do_prediction,
    probability,
    proba_binary,
    texture,
)


class SeagrassProcessor(Processor):
    def __init__(self, model, probability_threshold: int = 60, **kwargs):
        super().__init__(**kwargs)
        self._model = model
        self._probability_threshold = probability_threshold

    def process(self, input_data: xr.Dataset) -> xr.Dataset:
        scaled_data = scale(input_data).squeeze(drop=True)
        scaled_data = calculate_band_indices(scaled_data)
        masked_scaled, _ = all_masks(scaled_data, return_mask=True)
        texture_data = texture(masked_scaled.blue)
        # Getting dask errors without this compute, not sure why
        # But it probably makes sense, if we're not getting memory errors,
        # since do_prediction and probability probably each load into memory
        combined_data = xr.merge([masked_scaled, texture_data]).compute()
        no_data_value = 255
        classification = (
            do_prediction(combined_data, self._model)
            .fillna(no_data_value)
            .astype("uint8")
        )
        seagrass_code = 4
        seagrass_probability = probability(
            ds=combined_data,
            model=self._model,
            # or have the long list, danger here is they're out of order
            bands=list(combined_data.keys()),
            target_class_id=seagrass_code,
            # I _think_ this is the correct usage
            no_data_value=float("nan"),
        )
        seagrass_extent = proba_binary(seagrass_probability, 60, nodata_value=255)
        return xr.Dataset(
            {
                "classification": classification,
                "seagrass": seagrass_extent,
                # Do this now so it doesn't confuse proba_binary
                "seagrass_probability": seagrass_probability.fillna(
                    no_data_value
                ).astype("uint8"),
            }
        )
