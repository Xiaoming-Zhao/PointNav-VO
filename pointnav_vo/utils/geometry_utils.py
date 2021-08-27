#! /usr/bin/env python

import copy
import quaternion
import cv2
import pickle
import time
import numpy as np
from scipy.spatial.transform import Rotation

import torch

from habitat.core.simulator import AgentState
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_to_list,
    quaternion_rotate_vector,
)

from pointnav_vo.utils.rotation_utils import matrix_to_euler_angles
from pointnav_vo.utils.gaussian_blur import GaussianBlur2d


cv2.setNumThreads(0)


def quaternion_to_array(q: np.quaternion):
    r"""Creates coeffs in [x, y, z, w] format from quaternions
    """
    return np.array(quaternion_to_list(q), dtype=np.float32)


def modified_agent_state_target2ref(
    ref_agent_state: AgentState, target_agent_state: AgentState
) -> AgentState:
    r"""Computes the target agent_state's position and rotation representation
    with respect to the coordinate system defined by reference agent's position and rotation.
    :param ref_agent_state: reference agent_state,
        whose global position and rotation attributes define a local coordinate system.
    :param target_agent_state: target agent_state,
        whose global position and rotation attributes need to be transformed to
        the local coordinate system defined by ref_agent_state.
    """

    # [x, y, z, w] format
    delta_rotation = quaternion_to_array(
        ref_agent_state.rotation.inverse() * target_agent_state.rotation
    )

    delta_position = quaternion_rotate_vector(
        ref_agent_state.rotation.inverse(),
        target_agent_state.position - ref_agent_state.position,
    )

    return delta_rotation, delta_position


def quat_from_angle_axis(theta: float, axis: np.ndarray) -> np.quaternion:
    r"""Creates a quaternion from angle axis format
    :param theta: The angle to rotate about the axis by
    :param axis: The axis to rotate about
    :return: The quaternion
    """
    axis = axis.astype(np.float)
    axis /= np.linalg.norm(axis)
    return quaternion.from_rotation_vector(theta * axis)


def compute_global_state(prev_global_state, local_delta_state):
    r"""Compute current global state from
    - previous global state
    - state changes in local coordinate system defined by prev_global_state.

    Assume prev_global_rotation and cur_global_rotation as q1 and q2 respectively.
    Meanwhile, set prev_global_position and cur_global_position as v1 and v2.

    local_delta_pos = q_1^{-1} * (v2 - v1) * q_1
    ==> v2 = v1 + q_1 * local_delta_pos * q_1^{-1}

    local_delta_rot = q_1^{-1} * q_2
    ==> q_2 = q_1 * local_delta_rot

    :param prev_global_state: [rotation, position]
    :param local_delta_state: [dx, dz, dyaw]
    """
    prev_global_rot, prev_global_pos = prev_global_state
    dx, dz, dyaw = local_delta_state

    local_pos = np.array([dx, 0.0, dz])
    cur_global_pos = prev_global_pos + quaternion_rotate_vector(
        prev_global_rot, local_pos
    )

    local_delta_quaternion = quat_from_angle_axis(
        theta=dyaw, axis=np.array([0, 1.0, 0])
    )
    cur_global_rot = prev_global_rot * local_delta_quaternion

    return cur_global_rot, cur_global_pos


def get_polar_angle(agent_global_rotation):
    r"""Compute agent's heading in the coordinates of map.
    :param agent_global_rotation:quaternion is in [x, y, z, w] format
    """
    heading_vector = quaternion_rotate_vector(
        agent_global_rotation.inverse(), np.array([0, 0, -1])
    )

    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    x_y_flip = -np.pi / 2
    return np.array(phi) + x_y_flip


def compute_goal_pos(prev_goal_pos, local_delta_state):
    r"""Compute goal position w.r.t local coordinate system at time t + 1 from
    - goal position w.r.t. local coordinate system at time t 
    - state changes in local coordinate system at time t.

    Assume prev_goal_pos as prev_v_g.
    Meanwhile, set local_delta_pos and local_delta_rot as v and q respectively.

    cur_v_g = q^{-1} * (prev_v_g - v) * q

    :param prev_goal_pos: np.array
    :param local_delta_state: [dx, dz, dyaw]
    """
    dx, dz, dyaw = local_delta_state

    local_pos = np.array([dx, 0.0, dz])
    local_delta_quaternion = quat_from_angle_axis(
        theta=dyaw, axis=np.array([0, 1.0, 0])
    )
    cur_goal_pos = quaternion_rotate_vector(
        local_delta_quaternion.inverse(), prev_goal_pos - local_pos,
    )

    rho, phi = cartesian_to_polar(-cur_goal_pos[2], cur_goal_pos[0])

    out_dict = {
        "cartesian": cur_goal_pos,
        "polar": np.array([rho, -phi], dtype=np.float32),
    }
    return out_dict


def pointgoal_polar2catesian(pointgoal_polar):
    r"""Compute pointgoal position's catesian coordinates from polar coordinates.
    
    From https://github.com/facebookresearch/habitat-api/blob/d0db1b55be57abbacc5563dca2ca14654c545552/habitat/tasks/nav/nav.py#L171
    we know the pointgoal sensor is computed as:
    ```
    rho, phi = cartesian_to_polar(
        # x=-z, y=x
        -direction_vector_agent[2], direction_vector_agent[0]
    )
    return np.array([rho, -phi], dtype=np.float32)
    ```
    
    we just do the reverse computation.

    :param pointgoal_polar: np.array
    """

    rho = pointgoal_polar[0]
    phi = -1 * pointgoal_polar[1]

    # since the range of np.arctan2's image is [-pi, pi]
    # we could determine the sign of x and y
    if phi < 0:
        y = -1
    else:
        y = 1
    x = y / np.tan(phi)

    scale = rho / np.sqrt(x ** 2 + y ** 2)
    x *= scale
    y *= scale

    pointgoal_cat_x = y
    pointgoal_cat_z = -x

    return [pointgoal_cat_x, 0.0, pointgoal_cat_z]


def dyaw_to_rot_mat_np(dyaw):
    rot_mat = np.zeros((2, 2))

    rot_mat[0, 0] = np.cos(rad)
    rot_mat[0, 1] = np.sin(rad)
    rot_mat[1, 0] = -1 * np.sin(rad)
    rot_mat[1, 1] = np.cos(rad)

    return rot_mat


def dyaw_to_rot_mat_torch(dyaw: torch.Tensor) -> torch.Tensor:
    r"""Build a 2 x 2 rotation matrix from delta_yaw.
    
    Recall the formulation of 2-d rotation matrix is:
    [[ cos \theta,  0, sin \theta ],
     [ 0, 1, 0]
     [ -sin \theta, 0, cos \theta ]]
    
    Since we do not care about y-axis, we can directly remove the 2nd row and column.

    :param dyaw: torch.Tensor
    """

    rot_mat = torch.zeros((2, 2)).to(dyaw.device)

    rot_mat[0, 0] = torch.cos(rad)
    rot_mat[0, 1] = torch.sin(rad)
    rot_mat[1, 0] = -1 * torch.sin(rad)
    rot_mat[1, 1] = torch.cos(rad)

    return rot_mat


def rigid_transform_3D(A, B):

    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4767965
    # http://nghiaho.com/?page_id=671

    # assert len(A) == len(B)

    err_msg = ""

    A = copy.deepcopy(A)
    B = copy.deepcopy(B)

    num_rows, num_cols = A.shape

    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

    [num_rows, num_cols] = B.shape
    if num_rows != 3:
        raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3 x 1
    # (necessary when A or B are numpy arrays instead of numpy matrices)
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - np.tile(centroid_A, (1, num_cols))
    Bm = B - np.tile(centroid_B, (1, num_cols))

    H = np.matmul(Am, np.transpose(Bm))

    # sanity check
    if np.linalg.matrix_rank(H) < 3:
        err_msg = f"rank_{np.linalg.matrix_rank(H)}"
        return None, None, err_msg
        # raise ValueError("rank of H = {}, expecting 3".format(np.linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < 0, reflection detected!, correcting for it ...\n")
        Vt[2, :] *= -1
        R = np.matmul(Vt.T, U.T)

    t = -np.matmul(R, centroid_A) + centroid_B

    return R, t, err_msg


class NormalizedDepth2TopDownViewHabitat:
    def __init__(
        self,
        min_depth,
        max_depth,
        vis_size_h,
        vis_size_w,
        hfov_rad,
        ksize=3,
        rows_around_center=50,
        flag_center_crop=True,
    ):
        self._epsilon = 0.01

        self._min_depth = min_depth
        self._max_depth = max_depth
        self._vis_size_h = vis_size_h
        self._vis_size_w = vis_size_w
        self._hfov_rad = hfov_rad
        self._ksize = ksize
        self._rows_around_center = rows_around_center
        self._flag_center_crop = flag_center_crop

        self._get_intrinsic_mat()

    def gen_top_down_view(self, normalized_depth):
        # normalized_depth: [vis_size_h, vis_size_w, 1]
        depth_no_zero_border, depth_nonzero_infos = self._remove_depth_zero_border(
            normalized_depth
        )
        if depth_no_zero_border.size == 0:
            return np.zeros((self._vis_size_h, self._vis_size_w, 1))

        new_depth = cv2.GaussianBlur(
            depth_no_zero_border.astype(np.float32),
            (self._ksize, self._ksize),
            sigmaX=0,
            sigmaY=0,
            borderType=cv2.BORDER_ISOLATED,
        )

        # NOTE: DEBUG
        self._raw_depth = normalized_depth
        self._new_depth = new_depth

        coords_3d = self._compute_coords_3d(new_depth, depth_nonzero_infos[2])
        # new_coords_2d = pickle.loads(pickle.dumps(coords_3d[:2, :]))
        new_coords_2d = copy.deepcopy(coords_3d[:2, :])

        top_down_cnt, _ = self._cnt_points_in_pixel(new_coords_2d)

        cnt_list = top_down_cnt[top_down_cnt > 0].tolist()
        cnt_bound = np.max(cnt_list)

        if np.max(top_down_cnt) == 0:
            top_down_view = np.zeros((self._vis_size_h, self._vis_size_w))
        else:
            top_down_view = top_down_cnt / cnt_bound
            top_down_view[top_down_view > 1.0] = 1.0

        return top_down_view[..., np.newaxis]

    def _get_true_depth(self, depth):
        true_depth = depth * (self._max_depth - self._min_depth) + self._min_depth
        return true_depth

    def _get_intrinsic_mat(self):
        u0 = self._vis_size_w / 2
        v0 = self._vis_size_h / 2

        f = (self._vis_size_w / 2) / (np.tan(self._hfov_rad / 2))

        self._K = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1.0]])

    def _get_x_range(self, depth):
        # essestially, it is just np.tan(hfov_rad / 2) * depth
        homo_coords_2d = (self._vis_size_w - 0.5, 0, 1)  # right-most coordinate
        coords_3d = np.matmul(np.linalg.inv(self._K), homo_coords_2d) * depth
        return -coords_3d[0], coords_3d[0]

    def _remove_depth_zero_border(self, depth):
        for i in np.arange(depth.shape[0]):
            if np.sum(depth[i, :]) > 0:
                break
        min_row = i

        for i in np.arange(depth.shape[0] - 1, -1, -1):
            if np.sum(depth[i, :]) > 0:
                break
        max_row = i

        for j in np.arange(depth.shape[1]):
            if np.sum(depth[:, j]) > 0:
                break
        min_col = j

        for j in np.arange(depth.shape[1] - 1, -1, -1):
            if np.sum(depth[:, j]) > 0:
                break
        max_col = j

        return (
            depth[min_row : (max_row + 1), min_col : (max_col + 1), 0],
            (min_row, max_row, min_col, max_col),
        )

    def _compute_coords_3d(self, depth, min_nonzero_col):
        if self._flag_center_crop:
            # we select pixels around center horizontal line
            min_row = max(
                0, int(np.ceil(depth.shape[0] / 2)) - self._rows_around_center
            )
            max_row = min(
                depth.shape[0],
                int(np.ceil(depth.shape[0] / 2)) + self._rows_around_center,
            )
        else:
            min_row = 0
            max_row = min(self._rows_around_center * 2, depth.shape[0])

        valid_rows = max_row - min_row

        assert valid_rows <= depth.shape[0]

        # (u, v), u for horizontal, v for vertical
        v_coords, u_coords = np.meshgrid(
            np.arange(valid_rows), np.arange(depth.shape[1]), indexing="ij"
        )
        v_coords = v_coords.reshape(-1).astype(np.float16)
        u_coords = u_coords.reshape(-1).astype(np.float16) + min_nonzero_col

        # add 0.5 to generate 3D points from the center of pixels
        v_coords += 0.5
        u_coords += 0.5

        assert np.all(v_coords < self._vis_size_h)
        assert np.all(u_coords < self._vis_size_w)

        # [3, width * height]
        homo_coords_2d = np.array([u_coords, v_coords, np.ones(u_coords.shape)])

        coords_3d = np.matmul(np.linalg.inv(self._K), homo_coords_2d)
        assert np.all(coords_3d[-1, :] == 1)

        true_depth = self._get_true_depth(depth[min_row:max_row, :]).reshape(-1)
        coords_3d *= true_depth

        # change coordinate configuration from Habitat to normal one: positive-1st: right, positive-2nd: forward, postive-3rd: up
        coords_3d = coords_3d[[0, 2, 1], :]

        # the following is time-consuming
        """
        for i in range(coords_3d.shape[1]):
            tmp_min, tmp_max = get_x_range(coords_3d[1, i], hfov_rad, vis_size_h, vis_size_w)
            assert (
                coords_3d[0, i] >= tmp_min) and (coords_3d[0, i] <= tmp_max
            ), f"{i}, {v_coords[i]}, {u_coords[i]}, {coords_3d[2, i]}, {coords_3d[:, i]}, {tmp_min}, {tmp_max}, {np.tan(hfov_rad / 2) * coords_3d[2, i]}"
        """

        return coords_3d

    def _compute_pixel_coords(self, coords_2d):

        min_x, max_x = self._get_x_range(self._max_depth)
        x_range = max_x - min_x

        # normalize to [0, 1]
        ndc = coords_2d
        ndc[0, :] = (ndc[0, :] - min_x) / (x_range * (1 + self._epsilon))
        ndc[1, :] = (ndc[1, :] - self._min_depth) / (
            (self._max_depth - self._min_depth) * (1 + self._epsilon)
        )

        # assert np.all((ndc >= 0) & (ndc < 1)), f"{np.max(ndc[0, :])}, {np.max(coords_3d[0, :])}, {np.max(ndc[1, :])}, {np.max(coords_3d[1, :])}"

        # rescale to pixel
        # - in cartesian, origin locates at bottom-left, first element is for horizontal
        # - in image, origin locates at top-left, first element is for row
        pixel_coords = ndc[[1, 0], :]
        pixel_coords[0, :] = self._vis_size_h - np.ceil(
            self._vis_size_h * pixel_coords[0, :]
        )
        pixel_coords[1, :] = np.floor(self._vis_size_w * pixel_coords[1, :])
        # assert np.all(pixel_coords >= 0)
        pixel_coords = pixel_coords.astype(np.int)

        return pixel_coords

    def _cnt_points_in_pixel(self, coords_2d):
        pixel_coords = self._compute_pixel_coords(coords_2d)

        # unique_pixel_coords: [2, #]
        unique_pixel_coords, unique_cnt = np.unique(
            pixel_coords, axis=1, return_counts=True
        )

        top_down_cnt2 = np.zeros((self._vis_size_h, self._vis_size_w))

        flag1 = unique_pixel_coords[0, :] >= 0
        flag2 = unique_pixel_coords[0, :] < self._vis_size_h
        flag3 = unique_pixel_coords[1, :] >= 0
        flag4 = unique_pixel_coords[1, :] < self._vis_size_w
        # [#points, ]
        valid_flags = np.all(np.array((flag1, flag2, flag3, flag4)), axis=0)

        cnt_oob_points2 = unique_pixel_coords.shape[1] - np.sum(valid_flags)

        top_down_cnt2[
            unique_pixel_coords[0, valid_flags], unique_pixel_coords[1, valid_flags]
        ] = unique_cnt[valid_flags]

        return top_down_cnt2, cnt_oob_points2


class NormalizedDepth2TopDownViewHabitatTorch:
    def __init__(
        self,
        min_depth,
        max_depth,
        vis_size_h,
        vis_size_w,
        hfov_rad,
        ksize=3,
        rows_around_center=50,
        flag_center_crop=True,
    ):
        self._epsilon = 0.01

        self._min_depth = min_depth
        self._max_depth = max_depth
        self._vis_size_h = vis_size_h
        self._vis_size_w = vis_size_w
        self._hfov_rad = hfov_rad
        self._ksize = ksize
        self._rows_around_center = rows_around_center
        self._flag_center_crop = flag_center_crop

        self._get_intrinsic_mat()

    def gen_top_down_view(self, normalized_depth):

        # normalized_depth: [vis_size_h, vis_size_w, 1]
        depth_no_zero_border, depth_nonzero_infos = self._remove_depth_zero_border(
            normalized_depth
        )
        if torch.numel(depth_no_zero_border) == 0:
            return torch.zeros((self._vis_size_h, self._vis_size_w, 1)).to(
                normalized_depth.device
            )

        # [H, W]
        # Use OpenCV to avoid discrepancy with previous trained model
        new_depth = cv2.GaussianBlur(
            depth_no_zero_border.cpu().numpy(),
            (self._ksize, self._ksize),
            sigmaX=0,
            sigmaY=0,
            borderType=cv2.BORDER_ISOLATED,
        )
        new_depth = torch.FloatTensor(new_depth).to(normalized_depth.device)

        coords_3d = self._compute_coords_3d(new_depth, depth_nonzero_infos[2])
        new_coords_2d = coords_3d[:2, :].clone()

        top_down_cnt, _ = self._cnt_points_in_pixel(new_coords_2d)

        cnt_list = top_down_cnt[top_down_cnt > 0]
        cnt_bound = torch.max(top_down_cnt)

        if torch.max(top_down_cnt) == 0:
            # NOTE: we must have epsilon here since we center crop depth observations,
            # which will result in zero count if the depth is all black around center.
            top_down_view = torch.zeros((self._vis_size_h, self._vis_size_w)).to(
                normalized_depth.device
            )
        else:
            top_down_view = top_down_cnt / cnt_bound
            top_down_view[top_down_view > 1.0] = 1.0

        return top_down_view.unsqueeze(-1)

    def _get_true_depth(self, depth):
        true_depth = depth * (self._max_depth - self._min_depth) + self._min_depth
        return true_depth

    def _get_intrinsic_mat(self):
        u0 = self._vis_size_w / 2
        v0 = self._vis_size_h / 2

        f = (self._vis_size_w / 2) / (np.tan(self._hfov_rad / 2))

        self._K = torch.FloatTensor([[f, 0, u0], [0, f, v0], [0, 0, 1.0]])

    def _get_x_range(self, depth, device):
        # essestially, it is just np.tan(hfov_rad / 2) * depth
        homo_coords_2d = (self._vis_size_w - 0.5, 0, 1)  # right-most coordinate
        coords_3d = (
            torch.matmul(
                torch.inverse(self._K), torch.FloatTensor(homo_coords_2d).unsqueeze(-1)
            )
            * depth
        )
        coords_3d = coords_3d.to(device)
        return -coords_3d[0], coords_3d[0]

    def _remove_depth_zero_border(self, depth):
        for i in torch.arange(depth.shape[0]):
            if torch.sum(depth[i, :]) > 0:
                break
        min_row = i

        for i in torch.arange(depth.shape[0] - 1, -1, -1):
            if torch.sum(depth[i, :]) > 0:
                break
        max_row = i

        for j in torch.arange(depth.shape[1]):
            if torch.sum(depth[:, j]) > 0:
                break
        min_col = j

        for j in torch.arange(depth.shape[1] - 1, -1, -1):
            if torch.sum(depth[:, j]) > 0:
                break
        max_col = j

        return (
            depth[min_row : (max_row + 1), min_col : (max_col + 1), 0],
            (min_row, max_row, min_col, max_col),
        )

    def _compute_coords_3d(self, depth, min_nonzero_col):
        if self._flag_center_crop:
            # we select pixels around center horizontal line
            min_row = max(
                0, int(np.ceil(depth.shape[0] / 2)) - self._rows_around_center
            )
            max_row = min(
                depth.shape[0],
                int(np.ceil(depth.shape[0] / 2)) + self._rows_around_center,
            )
        else:
            min_row = 0
            max_row = min(self._rows_around_center * 2, depth.shape[0])

        valid_rows = max_row - min_row

        assert valid_rows <= depth.shape[0]

        # (u, v), u for horizontal, v for vertical
        v_coords, u_coords = torch.meshgrid(
            torch.arange(valid_rows).to(depth.device),
            torch.arange(depth.shape[1]).to(depth.device),
        )
        v_coords = v_coords.reshape(-1).float()  # .astype(np.float16)
        u_coords = (
            u_coords.reshape(-1).float() + min_nonzero_col
        )  # .astype(np.float16) + min_nonzero_col

        # add 0.5 to generate 3D points from the center of pixels
        v_coords += 0.5
        u_coords += 0.5

        assert torch.all(v_coords < self._vis_size_h)
        assert torch.all(u_coords < self._vis_size_w)

        # [3, width * height]
        homo_coords_2d = torch.stack(
            [u_coords, v_coords, torch.ones(u_coords.shape).to(depth.device)], dim=0
        )

        coords_3d = torch.matmul(
            torch.inverse(self._K).to(homo_coords_2d.device), homo_coords_2d
        )
        assert torch.all(coords_3d[-1, :] == 1)

        true_depth = self._get_true_depth(depth[min_row:max_row, :]).reshape(-1)

        coords_3d *= true_depth

        # change coordinate configuration from Habitat to normal one: positive-1st: right, positive-2nd: forward, postive-3rd: up
        coords_3d = coords_3d[[0, 2, 1], :]

        # the following is time-consuming
        """
        for i in range(coords_3d.shape[1]):
            tmp_min, tmp_max = get_x_range(coords_3d[1, i], hfov_rad, vis_size_h, vis_size_w)
            assert (
                coords_3d[0, i] >= tmp_min) and (coords_3d[0, i] <= tmp_max
            ), f"{i}, {v_coords[i]}, {u_coords[i]}, {coords_3d[2, i]}, {coords_3d[:, i]}, {tmp_min}, {tmp_max}, {np.tan(hfov_rad / 2) * coords_3d[2, i]}"
        """

        return coords_3d

    def _compute_pixel_coords(self, coords_2d):

        min_x, max_x = self._get_x_range(self._max_depth, device=coords_2d.device)
        x_range = max_x - min_x

        # normalize to [0, 1]
        ndc = coords_2d
        ndc[0, :] = (ndc[0, :] - min_x) / (x_range * (1 + self._epsilon))
        ndc[1, :] = (ndc[1, :] - self._min_depth) / (
            (self._max_depth - self._min_depth) * (1 + self._epsilon)
        )

        # rescale to pixel
        # - in cartesian, origin locates at bottom-left, first element is for horizontal
        # - in image, origin locates at top-left, first element is for row
        pixel_coords = ndc[[1, 0], :]
        pixel_coords[0, :] = self._vis_size_h - torch.ceil(
            self._vis_size_h * pixel_coords[0, :]
        )
        pixel_coords[1, :] = torch.floor(self._vis_size_w * pixel_coords[1, :])
        # assert np.all(pixel_coords >= 0)
        pixel_coords = pixel_coords.long()  # .astype(np.int)

        return pixel_coords

    def _cnt_points_in_pixel(self, coords_2d):

        pixel_coords = self._compute_pixel_coords(coords_2d)

        # unique_pixel_coords: [2, #]
        unique_pixel_coords, unique_cnt = torch.unique(
            pixel_coords, dim=1, sorted=False, return_counts=True
        )
        top_down_cnt = torch.zeros((self._vis_size_h, self._vis_size_w)).to(
            coords_2d.device
        )

        flag1 = unique_pixel_coords[0, :] >= 0
        flag2 = unique_pixel_coords[0, :] < self._vis_size_h
        flag3 = unique_pixel_coords[1, :] >= 0
        flag4 = unique_pixel_coords[1, :] < self._vis_size_w
        # [#points, ]
        valid_flags = torch.all(torch.stack((flag1, flag2, flag3, flag4), dim=0), dim=0)

        cnt_oob_points = unique_pixel_coords.shape[1] - torch.sum(valid_flags)

        top_down_cnt[
            unique_pixel_coords[0, valid_flags], unique_pixel_coords[1, valid_flags]
        ] = unique_cnt.float()[valid_flags]

        return top_down_cnt, cnt_oob_points


def disparity_to_depth(disparity, stereo_baseline, focal):
    return stereo_baseline * focal / (disparity.astype(np.float) + 1e-8)


def validate_rot_mat(R, eps=1e-6):
    Rt = np.transpose(R)
    expect_I = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - expect_I)
    ortho_mat = n < eps

    pos_det = np.abs(np.linalg.det(R) - 1) < eps

    return ortho_mat and pos_det


def get_relative_transform_from_mat(transform1, transform2):
    R1 = transform1[:3, :3]
    t1 = transform1[:, 3]
    assert validate_rot_mat(R1)

    R2 = transform2[:3, :3]
    t2 = transform2[:, 3]
    assert validate_rot_mat(R2)

    # [R_1 | t_1] [R_{1->2} | t_{1->2}] = [R_2 | t_2]
    # ==> [R_{1->2} | t_{1->2}] = [R_1^{-1} R_2 | R_1^{-1} (t_2 - t1)]

    rel_R = np.matmul(np.linalg.inv(R1), R2)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    rel_euler = Rotation.from_matrix(rel_R).as_euler("zyx", degrees=False)
    rel_t = np.matmul(np.linalg.inv(R1), (t2 - t1)[:, np.newaxis])[:, 0]

    # rot_mat = Rotation.from_euler('zyx', euler_angles, degrees=False).as_matrix()

    return rel_euler, rel_t


def validate_rot_mat_torch(R, eps=1e-6):
    Rt = torch.transpose(R, 0, 1)
    expect_I = torch.matmul(Rt, R)
    I = torch.eye(3)
    n = torch.norm(I - expect_I)
    ortho_mat = n < eps

    pos_det = np.abs(torch.det(R) - 1) < eps

    return ortho_mat and pos_det


def get_relative_transform_from_mat_torch(transform1, transform2):
    R1 = transform1[:3, :3]
    t1 = transform1[:, 3]
    assert validate_rot_mat_torch(R1)

    R2 = transform2[:3, :3]
    t2 = transform2[:, 3]
    assert validate_rot_mat_torch(R2)

    # [R_1 | t_1] [R_{1->2} | t_{1->2}] = [R_2 | t_2]
    # ==> [R_{1->2} | t_{1->2}] = [R_1^{-1} R_2 | R_1^{-1} (t_2 - t1)]

    rel_R = torch.matmul(torch.inverse(R1), R2)
    rel_euler = matrix_to_euler_angles(rel_R, convention="XYZ")
    rel_t = torch.matmul(torch.inverse(R1), (t2 - t1).unsqueeze(1))[:, 0]
    # transforms.rotation_conversions.euler_angles_to_matrix(angle_estimates_pytorch3d, convention="XYZ")

    return rel_euler, rel_t


def depth_map_to_3d_coords(unnormalized_depth, pixel_coords, K):
    """
    unnormalized_depth: [H, W]
    pixel_coords: [N, 2], 1st of coords is for row, 2nd of coords is for col
    K: intrinsic matrix
    """
    rows, cols = unnormalized_depth.shape

    # (u, v), u for horizontal, v for vertical
    v_coords = pixel_coords[:, 0].astype(np.float32)
    u_coords = pixel_coords[:, 1].astype(np.float32)

    # add 0.5 to generate 3D points from the center of pixels
    v_coords += 0.5
    u_coords += 0.5

    assert np.all(v_coords < rows), f"{np.max(v_coords)}, {rows}"
    assert np.all(u_coords < cols), f"{np.max(u_coords)}, {cols}"

    # [3, #points]
    homo_coords_2d = np.array([u_coords, v_coords, np.ones(u_coords.shape)])

    # [3, #points]
    coords_3d = np.matmul(np.linalg.inv(K), homo_coords_2d)
    assert np.all(coords_3d[-1, :] == 1)

    # [1, #points]
    depth_val = unnormalized_depth[pixel_coords[:, 0], pixel_coords[:, 1]].reshape(
        (1, u_coords.shape[0])
    )

    # [3, #points]
    coords_3d *= depth_val

    return coords_3d


def estimate_pose_by_essential_mat(kpts0, kpts1, K, thresh, conf=0.99999):
    """
    kpts0, kpts1: [N, 2]
    """

    if kpts0.shape[0] < 5:
        return None

    assert kpts0.dtype == np.float32 and kpts1.dtype == np.float32

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, K, threshold=thresh, prob=conf, method=cv2.RANSAC
    )

    # E, mask = cv2.findEssentialMat(
    #     kpts0, kpts1, focal=K[0, 0], pp=(K[0, 2], K[1, 2]), method=cv2.RANSAC, threshold=thresh, prob=conf)

    # R, t = 0, 0
    # _, R, t, _ = cv2.recoverPose(E, kpts0, kpts1, focal=K[0, 0], pp=(K[0, 2], K[1, 2]), mask=None)

    if E is None:
        return None
    else:
        best_num_inliers = 0
        ret = []
        for _E in np.split(E, E.shape[0] / 3):
            n, R, t, mask = cv2.recoverPose(_E, kpts0, kpts1, K, mask=mask)
            if n > best_num_inliers:
                best_num_inliers = n
                ret = [(n, R, t[:, 0], mask)]
            elif n == best_num_inliers:
                ret.append((n, R, t[:, 0], mask))
            else:
                pass
        return ret


def rel_pose_from_coord_change_to_camera_change(R_cur_rel_to_prev, t_cur_rel_to_prev):
    """
    This function computes cur_camera's position and orientation in prev_camera's coordinate system.
    
    coord_in_cur = R * coord_in_prev + t
    ==> coord_in_prev = R^{-1} * coord_in_cur - R^{-1} * t
    
    R_cur_rel_to_prev, t_cur_rel_to_prev: coordinates's basis change for relative pose.
    """
    R_prev_rel_to_cur = np.transpose(R_cur_rel_to_prev)
    t_prev_rel_to_cur = -1 * np.matmul(
        R_prev_rel_to_cur, t_cur_rel_to_prev.reshape((3, 1))
    )

    return R_prev_rel_to_cur, t_prev_rel_to_cur
