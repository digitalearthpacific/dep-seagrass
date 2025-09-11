from pathlib import Path
from zipfile import ZipFile

import boto3
import joblib
import requests
import typer
from dask.distributed import Client
from dep_tools.aws import object_exists
from dep_tools.exceptions import EmptyCollectionError
from dep_tools.grids import PACIFIC_GRID_10
from dep_tools.loaders import OdcLoader
from dep_tools.namers import S3ItemPath
from dep_tools.searchers import PystacSearcher
from dep_tools.stac_utils import StacCreator
from dep_tools.task import AwsStacTask as Task
from dep_tools.utils import get_logger
from odc.stac import configure_s3_access
from processor import SeagrassProcessor
from typing_extensions import Annotated


def main(
    tile_id: Annotated[str, typer.Option()],
    datetime: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    output_bucket: str = None,
    model: str = "classification/models/20250902c-alex.model",
    probability_threshold: int = 60,
    fast_mode: bool = True,
    xy_chunk_size: int = 1024,
    asset_url_prefix: str | None = None,
    decimated: bool = False,
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    log = get_logger(tile_id, "seagrass")
    log.info("Starting processing")

    tile_index = tuple(int(i) for i in tile_id.split(","))
    grid = PACIFIC_GRID_10
    geobox = grid.tile_geobox(tile_index)

    if decimated:
        geobox = geobox.zoom_out(10)

    # Make sure we can access S3
    log.info("Configuring S3 access")
    configure_s3_access(cloud_defaults=True)

    client = boto3.client("s3")

    # Download the model if we need to
    if model.startswith("https://"):
        model_local = "classification/models/" + model.split("/")[-1]
        if not Path(model_local).exists():
            log.info(f"Downloading model from {model} to {model_local}")
            r = requests.get(model)
            with open(model_local, "wb") as f:
                f.write(r.content)

        model = model_local

    if model.endswith(".zip"):
        unzipped = "classification/models/" + model.split("/")[-1].replace(".zip", "")
        if not Path(unzipped).exists():
            log.info("Unzipping model")
            with ZipFile(model, "r") as zip_ref:
                zip_ref.extractall(path="classification/models/")

        model = unzipped
        log.info(f"Unzipped model to {model}")

    # Make sure we can open the model now
    try:
        joblib.load(model)
    except Exception as e:
        log.exception(f"Failed to load model from {model}: {e}")
        typer.Exit(code=1)

    itempath = S3ItemPath(
        bucket=output_bucket,
        sensor="s2",
        dataset_id="seagrass",
        version=version,
        time=datetime,
        full_path_prefix=asset_url_prefix,
    )
    stac_url = itempath.stac_path(tile_id)

    # If we don't want to overwrite, and the destination file already exists, skip it
    if not overwrite and object_exists(output_bucket, stac_url, client=client):
        log.info(f"Item already exists at {itempath.stac_path(tile_id, absolute=True)}")
        # This is an exit with success
        raise typer.Exit()

    searcher = PystacSearcher(
        catalog="https://stac.digitalearthpacific.org",
        collections=["dep_s2_geomad"],
        datetime=datetime,
    )

    loader = OdcLoader(
        chunks=dict(x=xy_chunk_size, y=xy_chunk_size),
        fail_on_error=False,
        measurements=[
            "nir",
            "red",
            "blue",
            "green",
            "emad",
            "smad",
            "bcmad",
            "green",
            "nir08",
            "nir09",
            "swir16",
            "swir22",
            "coastal",
            "rededge1",
            "rededge2",
            "rededge3",
        ],  # List measurements so we don't get count
    )

    # The actual processor, doing the work :muscle:
    processor = SeagrassProcessor(
        model=joblib.load(model),
        probability_threshold=probability_threshold,
        nodata_value=255,
        fast_mode=fast_mode,
        log=log,
    )

    stac_creator = StacCreator(itempath=itempath, with_raster=True)

    try:
        client = Client(n_workers=4, threads_per_worker=16, memory_limit="12GB")
        log.info(("Started dask client"))
        paths = Task(
            itempath=itempath,
            id=tile_index,
            area=geobox,
            searcher=searcher,
            loader=loader,
            processor=processor,
            logger=log,
            stac_creator=stac_creator,
        ).run()
    except EmptyCollectionError:
        log.info("No items found for this tile")
        raise typer.Exit()  # Exit with success
    except Exception as e:
        log.exception(f"Failed to process with error: {e}")
        raise typer.Exit(code=1)
    finally:
        client.close()

    log.info(
        f"Completed processing. Wrote {len(paths)} items to {itempath.stac_path(tile_id, absolute=True)}"
    )


if __name__ == "__main__":
    typer.run(main)
