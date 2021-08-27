#! /usr/bin/env python

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnav_vo.utils.misc_utils import Flatten
from pointnav_vo.utils.baseline_registry import baseline_registry
from pointnav_vo.model_utils.visual_encoders import resnet
from pointnav_vo.model_utils.running_mean_and_var import RunningMeanAndVar
from pointnav_vo.vo.common.common_vars import *


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        *,
        observation_space,
        observation_size,
        baseplanes=32,
        ngroups=32,
        spatial_size_w=128,
        spatial_size_h=128,
        make_backbone=None,
        normalize_visual_inputs=False,
        after_compression_flat_size=2048,
        rgb_pair_channel=RGB_PAIR_CHANNEL,
        depth_pair_channel=DEPTH_PAIR_CHANNEL,
        discretized_depth_channels=0,
        top_down_view_pair_channel=TOP_DOWN_VIEW_PAIR_CHANNEL,
    ):
        super().__init__()

        if "rgb" in observation_space:
            self._n_input_rgb = rgb_pair_channel
            spatial_size_w, spatial_size_h = observation_size
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space:
            self._n_input_depth = depth_pair_channel
            spatial_size_w, spatial_size_h = observation_size
        else:
            self._n_input_depth = 0

        if "discretized_depth" in observation_space:
            spatial_size_w, spatial_size_h = observation_size
            self._n_input_discretized_depth = discretized_depth_channels * 2
        else:
            self._n_input_discretized_depth = 0

        if "top_down_view" in observation_space:
            spatial_size_w, spatial_size_h = observation_size
            self._n_input_top_down_view = top_down_view_pair_channel
        else:
            self._n_input_top_down_view = 0

        input_channels = (
            self._n_input_depth
            + self._n_input_rgb
            + self._n_input_discretized_depth
            + self._n_input_top_down_view
        )

        # NOTE: visual odometry must not be blind
        assert input_channels > 0

        if normalize_visual_inputs:
            self.running_mean_and_var = RunningMeanAndVar(input_channels)
        else:
            self.running_mean_and_var = nn.Sequential()

        self.backbone = make_backbone(input_channels, baseplanes, ngroups)
        final_spatial_w = int(
            np.ceil(spatial_size_w * self.backbone.final_spatial_compress)
        )
        final_spatial_h = int(
            np.ceil(spatial_size_h * self.backbone.final_spatial_compress)
        )
        num_compression_channels = int(
            round(after_compression_flat_size / (final_spatial_w * final_spatial_h))
        )
        self.compression = nn.Sequential(
            nn.Conv2d(
                self.backbone.final_channels,
                num_compression_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, num_compression_channels),
            nn.ReLU(True),
        )

        self.output_shape = (
            num_compression_channels,
            final_spatial_h,
            final_spatial_w,
        )

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observation_pairs):

        cnn_input = []

        if self._n_input_rgb > 0:
            rgb_observations = observation_pairs["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            # [prev_rgb, cur_rgb]
            cnn_input.append(
                [
                    rgb_observations[:, : self._n_input_rgb // 2, :],
                    rgb_observations[:, self._n_input_rgb // 2 :, :],
                ]
            )

        if self._n_input_depth > 0:
            depth_observations = observation_pairs["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(
                [
                    depth_observations[:, : self._n_input_depth // 2, :],
                    depth_observations[:, self._n_input_depth // 2 :, :],
                ]
            )

        if self._n_input_discretized_depth > 0:
            discretized_depth_observations = observation_pairs["discretized_depth"]
            discretized_depth_observations = discretized_depth_observations.permute(
                0, 3, 1, 2
            )
            cnn_input.append(
                [
                    discretized_depth_observations[
                        :, : self._n_input_discretized_depth // 2, :
                    ],
                    discretized_depth_observations[
                        :, self._n_input_discretized_depth // 2 :, :
                    ],
                ]
            )

        if self._n_input_top_down_view > 0:
            top_down_view_observations = observation_pairs["top_down_view"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            top_down_view_observations = top_down_view_observations.permute(0, 3, 1, 2)
            cnn_input.append(
                [
                    top_down_view_observations[
                        :, : self._n_input_top_down_view // 2, :
                    ],
                    top_down_view_observations[
                        :, self._n_input_top_down_view // 2 :, :
                    ],
                ]
            )

        # input order:
        # [prev_rgb, prev_depth, prev_discretized_depth, prev_top_down_view,
        #  cur_rgb, cur_depth, cur_discretized_depth, cur_top_down_view]
        cnn_input = [j for i in list(zip(*cnn_input)) for j in i]

        x = torch.cat(cnn_input, dim=1)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


class VisualOdometryCNNBase(nn.Module):
    def __init__(
        self,
        *,
        observation_space,
        observation_size,
        hidden_size=512,
        resnet_baseplanes=32,
        backbone="resnet18",
        normalize_visual_inputs=False,
        output_dim=DEFAULT_DELTA_STATE_SIZE,
        dropout_p=0.2,
        after_compression_flat_size=2048,
        rgb_pair_channel=RGB_PAIR_CHANNEL,
        depth_pair_channel=DEPTH_PAIR_CHANNEL,
        discretized_depth_channels=0,
        top_down_view_pair_channel=TOP_DOWN_VIEW_PAIR_CHANNEL,
    ):
        super().__init__()

        self.visual_encoder = ResNetEncoder(
            observation_space=observation_space,
            observation_size=observation_size,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
            after_compression_flat_size=after_compression_flat_size,
            rgb_pair_channel=rgb_pair_channel,
            depth_pair_channel=depth_pair_channel,
            discretized_depth_channels=discretized_depth_channels,
            top_down_view_pair_channel=top_down_view_pair_channel,
        )

        self.visual_fc = nn.Sequential(
            Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(np.prod(self.visual_encoder.output_shape), hidden_size),
            nn.ReLU(True),
        )

        self.output_head = nn.Sequential(
            nn.Dropout(dropout_p), nn.Linear(hidden_size, output_dim),
        )
        nn.init.orthogonal_(self.output_head[1].weight)
        nn.init.constant_(self.output_head[1].bias, 0)

    def forward(self, observation_pairs):
        visual_feats = self.visual_encoder(observation_pairs)
        visual_feats = self.visual_fc(visual_feats)
        output = self.output_head(visual_feats)
        return output


@baseline_registry.register_vo_model(name="vo_cnn")
class VisualOdometryCNN(VisualOdometryCNNBase):
    def __init__(
        self,
        *,
        observation_space,
        observation_size,
        hidden_size=512,
        resnet_baseplanes=32,
        backbone="resnet18",
        normalize_visual_inputs=False,
        output_dim=DEFAULT_DELTA_STATE_SIZE,
        dropout_p=0.2,
        discretized_depth_channels=0,
        top_down_view_pair_channel=TOP_DOWN_VIEW_PAIR_CHANNEL,
    ):
        assert backbone == "resnet18"
        assert discretized_depth_channels == 0
        assert "discretized_depth" not in observation_space
        assert "top_down_view" not in observation_space

        super().__init__(
            observation_space=observation_space,
            observation_size=observation_size,
            hidden_size=hidden_size,
            resnet_baseplanes=resnet_baseplanes,
            backbone=backbone,
            normalize_visual_inputs=normalize_visual_inputs,
            output_dim=output_dim,
            dropout_p=dropout_p,
        )


@baseline_registry.register_vo_model(name="vo_cnn_rgb")
class VisualOdometryCNNRGB(VisualOdometryCNNBase):
    def __init__(
        self,
        *,
        observation_space,
        observation_size,
        hidden_size=512,
        resnet_baseplanes=32,
        backbone="resnet18",
        normalize_visual_inputs=False,
        output_dim=DEFAULT_DELTA_STATE_SIZE,
        dropout_p=0.2,
        discretized_depth_channels=0,
        top_down_view_pair_channel=TOP_DOWN_VIEW_PAIR_CHANNEL,
    ):
        assert backbone == "resnet18"
        assert discretized_depth_channels == 0
        assert "depth" not in observation_space
        assert "discretized_depth" not in observation_space
        assert "top_down_view" not in observation_space

        super().__init__(
            observation_space=observation_space,
            observation_size=observation_size,
            hidden_size=hidden_size,
            resnet_baseplanes=resnet_baseplanes,
            backbone=backbone,
            normalize_visual_inputs=normalize_visual_inputs,
            output_dim=output_dim,
            dropout_p=dropout_p,
        )


@baseline_registry.register_vo_model(name="vo_cnn_wider")
class VisualOdometryCNNWider(VisualOdometryCNNBase):
    def __init__(
        self,
        *,
        observation_space,
        observation_size,
        hidden_size=512,
        resnet_baseplanes=32,
        backbone="resnet18",
        normalize_visual_inputs=False,
        output_dim=DEFAULT_DELTA_STATE_SIZE,
        dropout_p=0.2,
        discretized_depth_channels=0,
        top_down_view_pair_channel=TOP_DOWN_VIEW_PAIR_CHANNEL,
    ):
        assert backbone == "resnet18"
        assert discretized_depth_channels == 0
        assert "discretized_depth" not in observation_space
        assert "top_down_view" not in observation_space

        # Make the encoder 2x wide will result in ~3x #params
        resnet_baseplanes = 2 * resnet_baseplanes

        super().__init__(
            observation_space=observation_space,
            observation_size=observation_size,
            hidden_size=hidden_size,
            resnet_baseplanes=resnet_baseplanes,
            backbone=backbone,
            normalize_visual_inputs=normalize_visual_inputs,
            output_dim=output_dim,
            dropout_p=dropout_p,
        )


@baseline_registry.register_vo_model(name="vo_cnn_deeper")
class VisualOdometryCNNDeeper(VisualOdometryCNNBase):
    def __init__(
        self,
        *,
        observation_space,
        observation_size,
        hidden_size=512,
        resnet_baseplanes=32,
        backbone="resnet101",
        normalize_visual_inputs=False,
        output_dim=DEFAULT_DELTA_STATE_SIZE,
        dropout_p=0.2,
        discretized_depth_channels=0,
        top_down_view_pair_channel=TOP_DOWN_VIEW_PAIR_CHANNEL,
    ):

        assert backbone == "resnet101"
        assert discretized_depth_channels == 0
        assert "discretized_depth" not in observation_space
        assert "top_down_view" not in observation_space

        super().__init__(
            observation_space=observation_space,
            observation_size=observation_size,
            hidden_size=hidden_size,
            resnet_baseplanes=resnet_baseplanes,
            backbone=backbone,
            normalize_visual_inputs=normalize_visual_inputs,
            output_dim=output_dim,
            dropout_p=dropout_p,
        )


@baseline_registry.register_vo_model(name="vo_cnn_rgb_d_dd")
class VisualOdometryCNNDiscretizedDepth(VisualOdometryCNNBase):
    def __init__(
        self,
        *,
        observation_space,
        observation_size,
        hidden_size=512,
        resnet_baseplanes=32,
        backbone="resnet18",
        normalize_visual_inputs=False,
        output_dim=DEFAULT_DELTA_STATE_SIZE,
        dropout_p=0.2,
        discretized_depth_channels=10,
        top_down_view_pair_channel=TOP_DOWN_VIEW_PAIR_CHANNEL,
    ):

        assert backbone == "resnet18"
        assert "discretized_depth" in observation_space
        assert "top_down_view" not in observation_space

        super().__init__(
            observation_space=observation_space,
            observation_size=observation_size,
            hidden_size=hidden_size,
            resnet_baseplanes=resnet_baseplanes,
            backbone=backbone,
            normalize_visual_inputs=normalize_visual_inputs,
            output_dim=output_dim,
            dropout_p=dropout_p,
            discretized_depth_channels=discretized_depth_channels,
            after_compression_flat_size=2048,
        )


@baseline_registry.register_vo_model(name="vo_cnn_rgb_d_top_down")
class VisualOdometryCNN_RGB_D_TopDownView(VisualOdometryCNNBase):
    def __init__(
        self,
        *,
        observation_space,
        observation_size,
        hidden_size=512,
        resnet_baseplanes=32,
        backbone="resnet18",
        normalize_visual_inputs=False,
        output_dim=DEFAULT_DELTA_STATE_SIZE,
        dropout_p=0.2,
        discretized_depth_channels=10,
        top_down_view_pair_channel=TOP_DOWN_VIEW_PAIR_CHANNEL,
    ):

        assert backbone == "resnet18"
        assert "rgb" in observation_space
        assert "depth" in observation_space
        assert "discretized_depth" not in observation_space
        assert "top_down_view" in observation_space

        super().__init__(
            observation_space=observation_space,
            observation_size=observation_size,
            hidden_size=hidden_size,
            resnet_baseplanes=resnet_baseplanes,
            backbone=backbone,
            normalize_visual_inputs=normalize_visual_inputs,
            output_dim=output_dim,
            dropout_p=dropout_p,
            after_compression_flat_size=2048,
            top_down_view_pair_channel=top_down_view_pair_channel,
        )


@baseline_registry.register_vo_model(name="vo_cnn_rgb_dd_top_down")
class VisualOdometryCNN_RGB_DD_TopDownView(VisualOdometryCNNBase):
    def __init__(
        self,
        *,
        observation_space,
        observation_size,
        hidden_size=512,
        resnet_baseplanes=32,
        backbone="resnet18",
        normalize_visual_inputs=False,
        output_dim=DEFAULT_DELTA_STATE_SIZE,
        dropout_p=0.2,
        discretized_depth_channels=10,
        top_down_view_pair_channel=TOP_DOWN_VIEW_PAIR_CHANNEL,
    ):

        assert backbone == "resnet18"
        assert "rgb" in observation_space
        assert "depth" not in observation_space
        assert "discretized_depth" in observation_space
        assert "top_down_view" in observation_space

        super().__init__(
            observation_space=observation_space,
            observation_size=observation_size,
            hidden_size=hidden_size,
            resnet_baseplanes=resnet_baseplanes,
            backbone=backbone,
            normalize_visual_inputs=normalize_visual_inputs,
            output_dim=output_dim,
            dropout_p=dropout_p,
            after_compression_flat_size=2048,
            discretized_depth_channels=discretized_depth_channels,
            top_down_view_pair_channel=top_down_view_pair_channel,
        )


@baseline_registry.register_vo_model(name="vo_cnn_d_dd_top_down")
class VisualOdometryCNN_D_DD_TopDownView(VisualOdometryCNNBase):
    def __init__(
        self,
        *,
        observation_space,
        observation_size,
        hidden_size=512,
        resnet_baseplanes=32,
        backbone="resnet18",
        normalize_visual_inputs=False,
        output_dim=DEFAULT_DELTA_STATE_SIZE,
        dropout_p=0.2,
        discretized_depth_channels=10,
        top_down_view_pair_channel=TOP_DOWN_VIEW_PAIR_CHANNEL,
    ):

        assert backbone == "resnet18"
        assert "rgb" not in observation_space
        assert "depth" in observation_space
        assert "discretized_depth" in observation_space
        assert "top_down_view" in observation_space

        super().__init__(
            observation_space=observation_space,
            observation_size=observation_size,
            hidden_size=hidden_size,
            resnet_baseplanes=resnet_baseplanes,
            backbone=backbone,
            normalize_visual_inputs=normalize_visual_inputs,
            output_dim=output_dim,
            dropout_p=dropout_p,
            after_compression_flat_size=2048,
            discretized_depth_channels=discretized_depth_channels,
            top_down_view_pair_channel=top_down_view_pair_channel,
        )


@baseline_registry.register_vo_model(name="vo_cnn_rgb_d_dd_top_down")
class VisualOdometryCNNDiscretizedDepthTopDownView(VisualOdometryCNNBase):
    def __init__(
        self,
        *,
        observation_space,
        observation_size,
        hidden_size=512,
        resnet_baseplanes=32,
        backbone="resnet18",
        normalize_visual_inputs=False,
        output_dim=DEFAULT_DELTA_STATE_SIZE,
        dropout_p=0.2,
        discretized_depth_channels=10,
        top_down_view_pair_channel=TOP_DOWN_VIEW_PAIR_CHANNEL,
    ):

        assert backbone == "resnet18"
        assert "discretized_depth" in observation_space
        assert "top_down_view" in observation_space

        super().__init__(
            observation_space=observation_space,
            observation_size=observation_size,
            hidden_size=hidden_size,
            resnet_baseplanes=resnet_baseplanes,
            backbone=backbone,
            normalize_visual_inputs=normalize_visual_inputs,
            output_dim=output_dim,
            dropout_p=dropout_p,
            after_compression_flat_size=2048,
            discretized_depth_channels=discretized_depth_channels,
            top_down_view_pair_channel=top_down_view_pair_channel,
        )


@baseline_registry.register_vo_model(name="vo_cnn_discretize_depth_top_down")
class LegacyVisualOdometryCNNDiscretizedDepthTopDownView(
    VisualOdometryCNNDiscretizedDepthTopDownView
):
    pass
