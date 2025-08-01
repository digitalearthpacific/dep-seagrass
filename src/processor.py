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
        classification = do_prediction(combined_data, self._model)
        seagrass_code = 4
        seagrass_probability = probability(
            ds=combined_data,
            model=self._model,
            bands=list(combined_data.keys()),
            target_class_id=seagrass_code,
        )
        seagrass_extent = proba_binary(seagrass_probability, 60)
        return xr.Dataset(
            {
                "classification": classification,
                "seagrass": seagrass_extent,
                "seagrass_probability": seagrass_probability,
            }
        )
