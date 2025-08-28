from dep_tools.processors import Processor
import xarray as xr
from odc.algo import binary_erosion


from masking import all_masks
from utils import (
    calculate_band_indices,
    scale,
    do_prediction,
    probability,
    proba_binary,
    texture,
)


def finalise_format(da: xr.DataArray, no_data_value: int) -> xr.DataArray:
    """
    Fills NaN values, sets data type to uint8, and applies key geospatial attributes.
    """
    final_da = da.fillna(no_data_value).astype("uint8")
    final_da.attrs.update({"nodata": no_data_value, "_FillValue": no_data_value})
    return final_da


class SeagrassProcessor(Processor):
    def __init__(self, model, probability_threshold: int = 60, **kwargs):
        super().__init__(**kwargs)
        self._model = model
        self._probability_threshold = probability_threshold

    def process(self, input_data: xr.Dataset) -> xr.Dataset:
        scaled_data = scale(input_data).squeeze(drop=True)
        scaled_data = calculate_band_indices(scaled_data)
        masked_scaled, mask = all_masks(scaled_data, return_mask=True)
        # Eroding the texture mask by factor of (window size/2 i.e. 4.5 - need to test result)
        texture_mask = binary_erosion(mask, radius=4.5)
        texture_data = scaled_data.blue.where(texture_mask)
        texture_data = texture(texture_data)
        # Getting dask errors without this compute, not sure why
        # But it probably makes sense, if we're not getting memory errors,
        # since do_prediction and probability probably each load into memory
        combined_data = xr.merge([masked_scaled, texture_data]).compute()

        # Formatting function applied to combined_data
        classification = finalise_format(
            do_prediction(combined_data, self._model), no_data_value
        )

        seagrass_code = 4
        seagrass_probability = probability(
            ds=combined_data,
            model=self._model,
            # or have the long list, danger here is they're possibly out of order
            bands=list(combined_data.keys()),
            target_class_id=seagrass_code,
            # I _think_ this is the correct usage
            no_data_value=float("nan"),
            # Adding this section based on advice from JA around final output nodata attributes
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
