from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
)
import json
from shapely.geometry import shape
from shapely.geometry import Polygon, Point, box
import random
from sentinelhub import WebFeatureService, BBox, CRS, DataCollection, SHConfig
from datetime import datetime
from tqdm import tqdm
from sentinelhub import SHConfig


######Authentification#####
import os
CLIENT_ID_SENTINEL = os.getenv("CLIENT_ID_SENTINEL")
CLIENT_SECRET_SENTINEL = os.getenv("CLIENT_SECRET_SENTINEL")
INSTANCE_ID_SENTINEL = os.getenv("INSTANCE_ID_SENTINEL")

config = SHConfig(
    sh_client_id=CLIENT_ID_SENTINEL,
    sh_client_secret= CLIENT_SECRET_SENTINEL,
)

config.instance_id = INSTANCE_ID_SENTINEL

if not config.sh_client_id or not config.sh_client_secret:
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")


######Functions#####

def read_get_aoi_polygons(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    polygons = []
    for feature in data['features']:
        polygons.append(shape(feature['geometry']))
    return polygons


def plot_image(
    image: np.ndarray, factor: float = 1.0, clip_range: tuple[float, float] | None = None, **kwargs: Any
) -> None:
    """Utility function for plotting RGB images."""
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])


def generate_random_point_in_polygon(polygon):
  min_x, min_y, max_x, max_y = polygon.bounds
  while True:
    random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
    if polygon.contains(random_point):
      return random_point


def generate_random_bbox(polygon, bbox_size=0.01):
  random_point = generate_random_point_in_polygon(polygon)
  small_bbox=  BBox(bbox=( random_point.x - bbox_size / 2,
                          random_point.y - bbox_size / 2,
                           random_point.x + bbox_size / 2,
                           random_point.y + bbox_size / 2),
                     crs=CRS.WGS84)

  return small_bbox


def check_tiles_overlap(bbox,date_from,date_to,data_collection=DataCollection.SENTINEL2_L2A,cloud_coverage_max=0.5,config=config):
  wfs = WebFeatureService(
    bbox=bbox,
    time_interval=(date_from,date_to),
    data_collection=data_collection,
    config=config,
    maxcc=cloud_coverage_max,  # Max Cloud Cover %
    )
  tiles = wfs.get_tiles()
  dates = [x[1] for x in tiles]

  # print(dates)
  unique_dates = []
  for date in dates:
    if dates.count(date) == 1:
      unique_dates.append(date)
  return unique_dates


def tile_at_min_cc(bbox,date_from,date_to,data_collection=DataCollection.SENTINEL2_L2A,cloud_coverage_max=0.5,config=config):
  tile_dates = check_tiles_overlap(bbox,date_from,date_to,data_collection,cloud_coverage_max-0.5)

  while len(tile_dates)>0 and cloud_coverage_max >= 0:
    cloud_coverage_max = cloud_coverage_max - 0.05
    tile_dates = check_tiles_overlap(bbox,date_from,date_to,data_collection,cloud_coverage_max)

  return check_tiles_overlap(bbox,date_from,date_to,data_collection,cloud_coverage_max+0.05)