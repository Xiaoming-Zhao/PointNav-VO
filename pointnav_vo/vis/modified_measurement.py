"""
Modifications are based on https://github.com/facebookresearch/habitat-api/blob/d0db1b55be57abbacc5563dca2ca14654c545552/habitat/tasks/nav/nav.py#L664
"""

import numpy as np
import cv2
from copy import deepcopy
from typing import Any, Dict, List, Optional, Type, Union

from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.dataset import Episode
from habitat.core.embodied_task import Measure
from habitat.core.simulator import (
    AgentState,
    Simulator,
)
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    agent_state_target2ref,
    quaternion_rotate_vector,
)
from habitat.tasks.nav.nav import MAP_THICKNESS_SCALAR
from habitat.utils.visualizations import fog_of_war, maps
from pointnav_vo.vis.utils import (
    modified_get_topdown_map,
    modified_to_grid,
    COORDINATE_MIN,
    COORDINATE_MAX,
)


@registry.register_measure
class ModifiedTopDownMap(Measure):
    r"""Top Down Map measure
    """

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._grid_delta = config.MAP_PADDING
        self._step_count = None
        self._map_resolution = (config.MAP_RESOLUTION, config.MAP_RESOLUTION)
        self._num_samples = config.NUM_TOPDOWN_MAP_SAMPLE_POINTS
        self._ind_x_min = None
        self._ind_x_max = None
        self._ind_y_min = None
        self._ind_y_max = None
        self._previous_xy_location = None
        self._coordinate_min = COORDINATE_MIN
        self._coordinate_max = COORDINATE_MAX
        self._top_down_map = None
        self._shortest_path_points = None
        self._cell_scale = (
            self._coordinate_max - self._coordinate_min
        ) / self._map_resolution[0]
        self.line_thickness = int(
            np.round(self._map_resolution[0] * 2 / MAP_THICKNESS_SCALAR)
        )
        self.point_padding = 2 * int(
            np.ceil(self._map_resolution[0] / MAP_THICKNESS_SCALAR)
        )
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "top_down_map"

    def _check_valid_nav_point(self, point: List[float]):
        self._sim.is_navigable(point)

    def get_original_map(self):
        # top_down_map = maps.get_topdown_map(
        #     self._sim,
        #     self._map_resolution,
        #     self._num_samples,
        #     self._config.DRAW_BORDER,
        # )

        # use modifed map to get high-resolution top down map
        (
            top_down_map,
            self._coordinate_min,
            self._coordinate_max,
        ) = modified_get_topdown_map(
            self._sim,
            self._map_resolution,
            self._num_samples,
            self._config.DRAW_BORDER,
        )

        self._cell_scale = (
            self._coordinate_max - self._coordinate_min
        ) / self._map_resolution[0]
        # self.line_thickness = int(
        #     np.round(self._map_resolution[0] * 2 / MAP_THICKNESS_SCALAR)
        # )
        self.point_padding = 2 * int(
            np.ceil(self._map_resolution[0] / MAP_THICKNESS_SCALAR)
        )

        range_x = np.where(np.any(top_down_map, axis=1))[0]
        range_y = np.where(np.any(top_down_map, axis=0))[0]

        self._ind_x_min = range_x[0]
        self._ind_x_max = range_x[-1]
        self._ind_y_min = range_y[0]
        self._ind_y_max = range_y[-1]

        self.line_thickness = int((self._ind_x_max - self._ind_x_min) / 100)

        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = np.zeros_like(top_down_map)

        return top_down_map

    def _draw_point(self, position, point_type):
        t_x, t_y = modified_to_grid(
            position[0],
            position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        self._top_down_map[
            t_x - self.point_padding : t_x + self.point_padding + 1,
            t_y - self.point_padding : t_y + self.point_padding + 1,
        ] = point_type

    def _draw_goals_view_points(self, episode):
        if self._config.DRAW_VIEW_POINTS:
            for goal in episode.goals:
                try:
                    if goal.view_points is not None:
                        for view_point in goal.view_points:
                            self._draw_point(
                                view_point.agent_state.position,
                                maps.MAP_VIEW_POINT_INDICATOR,
                            )
                except AttributeError:
                    pass

    def _draw_goals_positions(self, episode):
        if self._config.DRAW_GOAL_POSITIONS:

            for goal in episode.goals:
                try:
                    self._draw_point(goal.position, maps.MAP_TARGET_POINT_INDICATOR)
                except AttributeError:
                    pass

    def _draw_goals_aabb(self, episode):
        if self._config.DRAW_GOAL_AABBS:
            for goal in episode.goals:
                try:
                    sem_scene = self._sim.semantic_annotations()
                    object_id = goal.object_id
                    assert int(sem_scene.objects[object_id].id.split("_")[-1]) == int(
                        goal.object_id
                    ), f"Object_id doesn't correspond to id in semantic scene objects dictionary for episode: {episode}"

                    center = sem_scene.objects[object_id].aabb.center
                    x_len, _, z_len = sem_scene.objects[object_id].aabb.sizes / 2.0
                    # Nodes to draw rectangle
                    corners = [
                        center + np.array([x, 0, z])
                        for x, z in [
                            (-x_len, -z_len),
                            (-x_len, z_len),
                            (x_len, z_len),
                            (x_len, -z_len),
                            (-x_len, -z_len),
                        ]
                    ]

                    map_corners = [
                        modified_to_grid(
                            p[0],
                            p[2],
                            self._coordinate_min,
                            self._coordinate_max,
                            self._map_resolution,
                        )
                        for p in corners
                    ]

                    maps.draw_path(
                        self._top_down_map,
                        map_corners,
                        maps.MAP_TARGET_BOUNDING_BOX,
                        self.line_thickness,
                    )
                except AttributeError:
                    pass

    def _draw_shortest_path(self, episode: Episode, agent_position: AgentState):
        if self._config.DRAW_SHORTEST_PATH:
            self._shortest_path_points = self._sim.get_straight_shortest_path_points(
                agent_position, episode.goals[0].position
            )
            self._shortest_path_points = [
                modified_to_grid(
                    p[0],
                    p[2],
                    self._coordinate_min,
                    self._coordinate_max,
                    self._map_resolution,
                )
                for p in self._shortest_path_points
            ]
            maps.draw_path(
                self._top_down_map,
                self._shortest_path_points,
                maps.MAP_SHORTEST_PATH_COLOR,
                self.line_thickness,
            )

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._step_count = 0
        self._metric = None
        self._top_down_map = self.get_original_map()
        agent_position = self._sim.get_agent_state().position
        a_x, a_y = modified_to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        self._previous_xy_location = (a_y, a_x)

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        # draw source and target parts last to avoid overlap
        self._draw_goals_view_points(episode)
        self._draw_goals_aabb(episode)
        self._draw_goals_positions(episode)

        self._draw_shortest_path(episode, agent_position)

        if self._config.DRAW_SOURCE:
            self._draw_point(episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR)

        # NOTE
        self._blank_top_down_map = deepcopy(self._top_down_map)
        self._prev_agent_state = self._sim.get_agent_state()

    def _ori_clip_map(self, _map):
        return _map[
            self._ind_x_min - self._grid_delta : self._ind_x_max + self._grid_delta,
            self._ind_y_min - self._grid_delta : self._ind_y_max + self._grid_delta,
        ]

    def _clip_map(self, _map):
        return _map[
            max(self._ind_x_min - self._grid_delta, 0) : self._ind_x_max
            + self._grid_delta,
            max(self._ind_y_min - self._grid_delta, 0) : self._ind_y_max
            + self._grid_delta,
        ]

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        self._step_count += 1
        house_map, map_agent_x, map_agent_y = self.update_map(
            self._sim.get_agent_state().position
        )

        # Rather than return the whole map which may have large empty regions,
        # only return the occupied part (plus some padding).
        clipped_house_map = self._clip_map(house_map)

        clipped_fog_of_war_map = None
        if self._config.FOG_OF_WAR.DRAW:
            clipped_fog_of_war_map = self._clip_map(self._fog_of_war_mask)

        # modification
        self._cur_agent_state = self._sim.get_agent_state()

        prev_agent_angle = self.get_polar_angle(self._prev_agent_state)

        prev_state = (self._prev_agent_state.rotation, self._prev_agent_state.position)
        cur_state = (self._cur_agent_state.rotation, self._cur_agent_state.position)
        # (rotation, position)
        delta_state = agent_state_target2ref(prev_state, cur_state)

        self._prev_agent_state = self._cur_agent_state

        dx, _, dz = delta_state[1]
        dyaw = 2 * np.arctan2(delta_state[0].imag[1], delta_state[0].real)
        extra_infos = {
            "prev_state": {"rotation": prev_state[0], "position": prev_state[1]},
            "cur_state": {"rotation": cur_state[0], "position": cur_state[1]},
            "delta_position": delta_state[1],
            "delta_rotation": delta_state[0],
            "delta": [dx, dz, dyaw],
            "prev_agent_angle": prev_agent_angle,
            "map_infos": {
                "blank_top_down_map": self._clip_map(self._blank_top_down_map),
                "ind_x_min": self._ind_x_min,
                "ind_x_max": self._ind_x_max,
                "ind_y_min": self._ind_y_min,
                "ind_y_max": self._ind_y_max,
                "grid_delta": self._grid_delta,
                "coordinate_min": self._coordinate_min,
                "coordinate_max": self._coordinate_max,
                "map_resolution": self._map_resolution,
                "line_thickness": self.line_thickness,
            },
        }

        self._metric = {
            "map": clipped_house_map,
            "fog_of_war_mask": clipped_fog_of_war_map,
            "agent_map_coord": (
                map_agent_x - (self._ind_x_min - self._grid_delta),
                map_agent_y - (self._ind_y_min - self._grid_delta),
            ),
            "agent_angle": self.get_polar_angle(),
            # NOTE
            "extra_infos": extra_infos,
        }

    def get_polar_angle(self, agent_state=None):
        if agent_state is None:
            agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        x_y_flip = -np.pi / 2
        return np.array(phi) + x_y_flip

    def update_map(self, agent_position):
        a_x, a_y = modified_to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        # Don't draw over the source point
        if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
            color = 10 + min(
                self._step_count * 245 // self._config.MAX_EPISODE_STEPS, 245
            )

            # thickness = int(
            #     np.round(self._map_resolution[0] * 2 / MAP_THICKNESS_SCALAR)
            # )
            thickness = self.line_thickness
            cv2.line(
                self._top_down_map,
                self._previous_xy_location,
                (a_y, a_x),
                color,
                thickness=thickness,
            )

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        self._previous_xy_location = (a_y, a_x)
        return self._top_down_map, a_x, a_y

    def update_fog_of_war_mask(self, agent_position):
        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                self._fog_of_war_mask,
                agent_position,
                self.get_polar_angle(),
                fov=self._config.FOG_OF_WAR.FOV,
                max_line_len=self._config.FOG_OF_WAR.VISIBILITY_DIST
                * max(self._map_resolution)
                / (self._coordinate_max - self._coordinate_min),
            )
