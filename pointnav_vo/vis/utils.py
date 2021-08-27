#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import shutil
import imageio
import quaternion
import numpy as np
from typing import List, Optional, Tuple

import habitat
from habitat.core.simulator import Simulator
from habitat.utils.visualizations import maps
from habitat.tasks.utils import cartesian_to_polar

if habitat.__version__ > "0.1.5":
    FLAG_OLD = False
else:
    FLAG_OLD = True


# https://github.com/facebookresearch/habitat-lab/blob/d0db1b5/habitat/utils/visualizations/maps.py
COORDINATE_EPSILON = 1e-6
COORDINATE_MIN = -62.3241 - COORDINATE_EPSILON
COORDINATE_MAX = 90.0399 + COORDINATE_EPSILON


def resize_top_down_map(top_down_map, map_agent_info_list, output_size):
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array((1, original_map_size[1] * 1.0 / original_map_size[0]))
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    # map_agent_pos = info["top_down_map"]["agent_map_coord"]
    for map_agent_info in map_agent_info_list:
        map_agent_pos, heading = map_agent_info
        map_agent_pos = np.round(
            map_agent_pos * new_map_size / original_map_size
        ).astype(np.int32)
        top_down_map = maps.draw_agent(
            top_down_map,
            map_agent_pos,
            # heading - np.pi / 2,
            heading,
            agent_radius_px=top_down_map.shape[0] / 40,
        )
    return top_down_map


def modified_get_topdown_map(
    sim: Simulator,
    map_resolution: Tuple[int, int] = (1250, 1250),
    num_samples: int = 20000,
    draw_border: bool = True,
) -> np.ndarray:
    r"""Return a top-down occupancy map for a sim. Note, this only returns valid
    values for whatever floor the agent is currently on.
    Args:
        sim: The simulator.
        map_resolution: The resolution of map which will be computed and
            returned.
        num_samples: The number of random navigable points which will be
            initially
            sampled. For large environments it may need to be increased.
        draw_border: Whether to outline the border of the occupied spaces.
    Returns:
        Image containing 0 if occupied, 1 if unoccupied, and 2 if border (if
        the flag is set).
    """
    top_down_map = np.zeros(map_resolution, dtype=np.uint8)
    border_padding = 3

    start_height = sim.get_agent_state().position[1]

    cur_coordinate_min = -1 * COORDINATE_MIN
    cur_coordinate_max = -1 * COORDINATE_MAX

    # Use sampling to find the extrema points that might be navigable.
    range_x = (map_resolution[0], 0)
    range_y = (map_resolution[1], 0)

    point_list = []
    grid_point_list = []

    for _ in range(num_samples):
        point = sim.sample_navigable_point()
        # Check if on same level as original
        if np.abs(start_height - point[1]) > 0.5:
            continue

        point_list.append(point)

        cur_coordinate_min = min([cur_coordinate_min, point[0], point[2]])
        cur_coordinate_max = max([cur_coordinate_max, point[0], point[2]])

    for i in range(len(point_list)):
        point = point_list[i]
        if FLAG_OLD:
            g_x, g_y = maps.to_grid(
                point[0],
                point[2],
                cur_coordinate_min,
                cur_coordinate_max,
                map_resolution,
            )
        else:
            g_x, g_y = maps.to_grid(point[0], point[2], map_resolution, sim=sim,)
        range_x = (min(range_x[0], g_x), max(range_x[1], g_x))
        range_y = (min(range_y[0], g_y), max(range_y[1], g_y))

    # Pad the range just in case not enough points were sampled to get the true
    # extrema.
    padding = int(np.ceil(map_resolution[0] / 125))
    range_x = (
        max(range_x[0] - padding, 0),
        min(range_x[-1] + padding + 1, top_down_map.shape[0]),
    )
    range_y = (
        max(range_y[0] - padding, 0),
        min(range_y[-1] + padding + 1, top_down_map.shape[1]),
    )

    # Search over grid for valid points.
    for ii in range(range_x[0], range_x[1]):
        for jj in range(range_y[0], range_y[1]):
            if FLAG_OLD:
                realworld_x, realworld_y = maps.from_grid(
                    ii, jj, cur_coordinate_min, cur_coordinate_max, map_resolution
                )
            else:
                realworld_x, realworld_y = maps.from_grid(
                    ii, jj, map_resolution, sim=sim,
                )
            valid_point = sim.is_navigable([realworld_x, start_height, realworld_y])
            top_down_map[ii, jj] = (
                maps.MAP_VALID_POINT if valid_point else maps.MAP_INVALID_POINT
            )

    # Draw border if necessary
    if draw_border:
        # Recompute range in case padding added any more values.
        range_x = np.where(np.any(top_down_map, axis=1))[0]
        range_y = np.where(np.any(top_down_map, axis=0))[0]
        range_x = (
            max(range_x[0] - border_padding, 0),
            min(range_x[-1] + border_padding + 1, top_down_map.shape[0]),
        )
        range_y = (
            max(range_y[0] - border_padding, 0),
            min(range_y[-1] + border_padding + 1, top_down_map.shape[1]),
        )

        maps._outline_border(
            top_down_map[range_x[0] : range_x[1], range_y[0] : range_y[1]]
        )

    return top_down_map, cur_coordinate_min, cur_coordinate_max


def modified_to_grid(
    realworld_x: float,
    realworld_y: float,
    coordinate_min: float,
    coordinate_max: float,
    grid_resolution: Tuple[int, int],
) -> Tuple[int, int]:
    r"""Return gridworld index of realworld coordinates assuming top-left corner
    is the origin. The real world coordinates of lower left corner are
    (coordinate_min, coordinate_min) and of top right corner are
    (coordinate_max, coordinate_max)
    """
    grid_size = (
        (coordinate_max - coordinate_min) / grid_resolution[0],
        (coordinate_max - coordinate_min) / grid_resolution[1],
    )
    grid_x = min(
        int((coordinate_max - realworld_x) / grid_size[0]), grid_resolution[0] - 1
    )
    grid_y = min(
        int((realworld_y - coordinate_min) / grid_size[1]), grid_resolution[1] - 1
    )
    return grid_x, grid_y


def global_pos_to_map_coord(
    realworld_x: float, realworld_y: float, map_infos: dict,
) -> Tuple[int, int]:

    coordinate_min = map_infos["coordinate_min"]
    coordinate_max = map_infos["coordinate_max"]
    ind_x_min = map_infos["ind_x_min"]
    ind_y_min = map_infos["ind_y_min"]
    grid_delta = map_infos["grid_delta"]
    grid_resolution = map_infos["map_resolution"]

    map_x, map_y = modified_to_grid(
        realworld_x, realworld_y, coordinate_min, coordinate_max, grid_resolution,
    )

    map_coord = (
        map_x - (ind_x_min - grid_delta),
        map_y - (ind_y_min - grid_delta),
    )

    return map_coord
