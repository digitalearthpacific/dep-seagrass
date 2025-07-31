import boto3
import dask
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
from typing_extensions import Annotated

S2_BANDS = [
    "scl",
    "coastal",
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "nir08",
    "nir09",
    "swir16",
    "swir22",
]


def main(
    tile_id: Annotated[str, typer.Option()],
    datetime: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    output_bucket: str = None,
    base_product: str = "ls",
    memory_limit: str = "50GB",
    n_workers: int = 2,
    threads_per_worker: int = 32,
    xy_chunk_size: int = 3201,
    decimated: bool = False,
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    log = get_logger(tile_id)
    log.info("Starting processing.")

    tile_index = tuple(int(i) for i in tile_id.split(","))
    grid = PACIFIC_GRID_10
    geobox = grid.tile_geobox(tile_index)

    # Jesse: not sure what this is
    if decimated:
        log.warning("Running at 1/10th resolution")
        geobox = geobox.zoom_out(10)

    # Make sure we can access S3
    log.info("Configuring S3 access")
    configure_s3_access(cloud_defaults=True)

    client = boto3.client("s3")

    itempath = S3ItemPath(
        bucket=output_bucket,
        sensor=base_product,
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
        **search_kwargs,
    )

    # Not sure these need to be listed because I think it's everything
    measurements = [
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
    ]

    chunks = dict(x=xy_chunk_size, y=xy_chunk_size)

    loader = OdcLoader(
        bands=measurements,
        chunks=chunks,
        fail_on_error=False,
    )

    processor = Processor()

    # STAC making thing
    stac_creator = StacCreator(
        itempath=itempath, remote=True, make_hrefs_https=True, with_raster=True
    )

    try:
        # TODO: Shift dask config out to environment variables...
        with dask.config.set(
            {
                "dataframe.shuffle.method": "p2p",
                "distributed.worker.memory.target": False,
                "distributed.worker.memory.spill": False,
                "distributed.worker.memory.pause": 0.9,
                "distributed.worker.memory.terminate": 0.98,
            }
        ):
            with Client(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                memory_limit=memory_limit,
            ):
                log.info(
                    (
                        f"Started dask client with {n_workers} workers "
                        f"and {threads_per_worker} threads with "
                        f"{memory_limit} memory"
                    )
                )
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
        f"Completed processing. Wrote {len(paths)} items to https://{output_bucket}.s3.us-west-2.amazonaws.com/{ stac_document}"
    )


if __name__ == "__main__":
    typer.run(main)
