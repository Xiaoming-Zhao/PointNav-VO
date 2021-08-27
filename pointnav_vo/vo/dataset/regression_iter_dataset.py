#! /usr/bin/env python

import os
import h5py
import random
import time
import copy
import numpy as np
from tqdm import tqdm
from collections import OrderedDict, defaultdict

import torch
from torch.utils.data import Dataset, IterableDataset

from habitat import logger


class BaseRegressionDataset(IterableDataset):
    r"""Data loader for state-pairs from Habitat simularo.
    """

    def __init__(self):
        self._num_workers = 1
        self._transform_normalize = False
        self._transform_mean = None
        self._transform_std = None

    @property
    def transform_normalize(self):
        return self._transform_normalize

    def _discretize_depth_func(self, idx, raw_depth):
        assert np.max(raw_depth) <= 1.0
        assert np.min(raw_depth) >= 0.0

        # we only need {0, 1}
        discretized_depth = np.zeros(
            (*raw_depth.shape, self._discretized_depth_channels), dtype=np.uint8
        )

        for i in np.arange(self._discretized_depth_channels):
            if i == self._discretized_depth_channels - 1:
                # include the last end values
                pos = np.unravel_index(
                    np.flatnonzero(
                        (
                            (raw_depth >= self._discretized_depth_end_vals[i])
                            & (raw_depth <= self._discretized_depth_end_vals[i + 1])
                        )
                    ),
                    raw_depth.shape,
                )
            else:
                pos = np.unravel_index(
                    np.flatnonzero(
                        (
                            (raw_depth >= self._discretized_depth_end_vals[i])
                            & (raw_depth < self._discretized_depth_end_vals[i + 1])
                        )
                    ),
                    raw_depth.shape,
                )

            discretized_depth[pos[0], pos[1], i] = 1.0

        if self._discretize_depth == "hard":
            assert np.sum(discretized_depth) == raw_depth.size

        return discretized_depth
