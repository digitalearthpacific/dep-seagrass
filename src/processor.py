from dep_tools.processors import Processor
import xarray as xr

from masking import all_masks
from utils import scale, probability, proba_binary, texture


class SeagrassProcessor(Processor):
    def __init__(self, model, probability_threshold: int = 60, **kwargs):
        super().__init__(**kwargs)
        self._model = model
        self._probability_threshold = probability_threshold

    def process(self, data: xr.Dataset) -> xr.Dataset:
        scaled_data = scale(data).squeeze(drop=True)
        texture_data = texture(data.blue)
        breakpoint() # <- haven't got here yet
        combined_ds = xr.merge([scaled_data, texture_data])
        masked_scaled, mask = all_masks(combined_ds, return_mask=True)
        # notebook 4 gets confusing here. I'm not sure what is desired
        # I think it's this?
        seagrass_value = 4  # ???????
        probability_output = probability(
            masked_scaled,
            self._model,
            [
                "nir",
                "red",
                "blue",
                "green",
                "emad",
                "smad",
                "bcmad",
                "nir08",
                "nir09",
                "swir16",
                "swir22",
                "coastal",
                "rededge1",
                "rededge2",
                "rededge3",
                "mndwi",
                "ndti",
                "cai",
                "ndvi",
                "evi",
                "savi",
                "ndwi",
                "b_g",
                "b_r",
                "mci",
                "ndci",
                "ln_bg",
                "contrast",
                "homogeneity",
                "energy",
                "ASM",
                "correlation",
                "mean",
                "entropy",
            ],
            seagrass_value,
        )
        seagrass_extent = proba_binary(probability_output, 60)
        return xr.Dataset(
            {"seagrass": seagrass_extent, "seagrass_probability": probability_output}
        )
