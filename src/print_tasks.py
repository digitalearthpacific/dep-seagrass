import json
import sys
from itertools import product
from typing import Annotated, Optional

import boto3
import typer
from dep_tools.aws import object_exists
from dep_tools.grids import get_tiles
from dep_tools.namers import S3ItemPath
from dep_tools.parsers import datetime_parser

DATASET_ID = "seagrass"


def main(
    years: Annotated[str, typer.Option(parser=datetime_parser)],
    version: Annotated[str, typer.Option()],
    regions: Optional[str] = "ALL",
    tile_buffer_kms: Optional[int] = 0.0,
    limit: Optional[str] = None,
    base_product: str = "s2",
    output_bucket: Optional[str] = None,
    output_prefix: Optional[str] = None,
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    country_codes = None if regions.upper() == "ALL" else regions.split(",")

    tiles = get_tiles(
        country_codes=country_codes, buffer_distance=tile_buffer_kms * 1000
    )

    if limit is not None:
        limit = int(limit)

    tasks = [
        {
            "tile-id": ",".join([str(i) for i in tile[0]]),
            "year": year,
            "version": version,
        }
        for tile, year in product(list(tiles), years)
    ]

    # If we don't want to overwrite, then we should only run tasks that don't already exist
    # i.e., they failed in the past or they're missing for some other reason
    if not overwrite:
        valid_tasks = []
        client = boto3.client("s3")
        for task in tasks:
            itempath = S3ItemPath(
                bucket=output_bucket,
                sensor=base_product,
                dataset_id=DATASET_ID,
                version=version,
                time=task["year"],
            )
            stac_path = itempath.stac_path(task["tile-id"].split(","))

            if output_prefix is not None:
                stac_path = f"{output_prefix}/{stac_path}"

            if not object_exists(output_bucket, stac_path, client=client):
                valid_tasks.append(task)
            if len(valid_tasks) == limit:
                break
        # Switch to this list of tasks, which has been filtered
        tasks = valid_tasks

    if limit is not None:
        tasks = tasks[0:limit]

    json.dump(tasks, sys.stdout)


if __name__ == "__main__":
    typer.run(main)
