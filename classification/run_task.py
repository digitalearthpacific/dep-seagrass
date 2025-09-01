import boto3
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
import joblib
from odc.stac import configure_s3_access
from typing_extensions import Annotated

from processor import SeagrassProcessor


def main(
    tile_id: Annotated[str, typer.Option()],
    datetime: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    output_bucket: str = None,
    model: str = "classification/models/nm-27072025-test.model",
    probability_threshold: int = 60,
    fast_mode: bool = True,
    xy_chunk_size: int = 1024,
    asset_url_prefix: str | None = None,
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    log = get_logger(tile_id, "seagrass")
    log.info("Starting processing")

    tile_index = tuple(int(i) for i in tile_id.split(","))
    grid = PACIFIC_GRID_10
    geobox = grid.tile_geobox(tile_index)

    # Make sure we can access S3
    log.info("Configuring S3 access")
    configure_s3_access(cloud_defaults=True)

    client = boto3.client("s3")

    itempath = S3ItemPath(
        bucket=output_bucket,
        sensor="s2",
        dataset_id="seagrass",
        version=version,
        time=datetime,
    )
    stac_document = itempath.stac_path(tile_id)

    # If we don't want to overwrite, and the destination file already exists, skip it
    if not overwrite and object_exists(output_bucket, stac_document, client=client):
        log.info(f"Item already exists at {stac_document}")
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

    stac_creator = StacCreator(
        itempath=itempath,
        remote=True,
        make_hrefs_https=True,
        with_raster=True,
        asset_url_prefix=asset_url_prefix,
    )

    try:
        with Client(n_workers=4, threads_per_worker=16, memory_limit="8GB"):
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

    log.info(
        f"Completed processing. Wrote {len(paths)} items to {stac_creator.stac_url(tile_id)}"
    )


if __name__ == "__main__":
    typer.run(main)
