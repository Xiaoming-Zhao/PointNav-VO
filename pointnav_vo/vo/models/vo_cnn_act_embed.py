#! /usr/bin/env python


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnav_vo.utils.misc_utils import Flatten
from pointnav_vo.utils.baseline_registry import baseline_registry
from pointnav_vo.model_utils.visual_encoders import resnet
from .vo_cnn import ResNetEncoder
from pointnav_vo.vo.common.common_vars import *


@baseline_registry.register_vo_model(name="vo_cnn_act_embed")
class VisualOdometryCNNActEmbed(nn.Module):
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
        after_compression_flat_size=2048,
        n_acts=N_ACTS,
    ):
        super().__init__()

        self.action_embedding = nn.Embedding(n_acts + 1, EMBED_DIM)

        self.visual_encoder = ResNetEncoder(
            observation_space=observation_space,
            observation_size=observation_size,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
            discretized_depth_channels=discretized_depth_channels,
            after_compression_flat_size=after_compression_flat_size,
        )

        self.flatten = Flatten()

        self.hidden_generator = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(
                np.prod(self.visual_encoder.output_shape) + EMBED_DIM, hidden_size
            ),
            nn.ReLU(True),
        )

        self.output_head = nn.Sequential(
            nn.Dropout(dropout_p), nn.Linear(hidden_size, output_dim),
        )
        nn.init.orthogonal_(self.output_head[1].weight)
        nn.init.constant_(self.output_head[1].bias, 0)

    def forward(self, observation_pairs, actions):
        # [batch, embed_dim]
        act_embed = self.action_embedding(actions)
        encoder_output = self.visual_encoder(observation_pairs)
        visual_feats = self.flatten(encoder_output)

        all_feats = torch.cat((visual_feats, act_embed), dim=1)
        hidden_feats = self.hidden_generator(all_feats)

        output = self.output_head(hidden_feats)
        return output


@baseline_registry.register_vo_model(name="vo_cnn_wider_act_embed")
class VisualOdometryCNNWiderActEmbed(VisualOdometryCNNActEmbed):
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
        n_acts=N_ACTS,
        discretized_depth_channels=0,
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
            n_acts=n_acts,
        )
