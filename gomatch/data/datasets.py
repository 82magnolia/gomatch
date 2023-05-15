from argparse import Namespace
from typing import Any, Dict, Mapping, Sequence, Union
import os

import numpy as np
import torch.utils.data as data
import yaml

from .data_processing import (
    subsample_points_indices,
    compute_gt_2d3d_match,
    generate_assignment_mask,
)
from glob import glob


class BaseDataset(data.Dataset):
    default_config = dict(
        data_root="data/matterport_kpts",
        dataset="mp3d",
        feat_type="sphere",
        npts=[10, 1024],
        outlier_rate=[0, 1],
        inls2d_thres=0.1,
        normalized_thres=True,
    )

    def __init__(
        self, config: Union[Namespace, Mapping[str, Any]], split: str = "train"
    ) -> None:
        if isinstance(config, Namespace):
            config = vars(config)
        config = Namespace(**{**self.default_config, **config})
        self.feat_type = config.feat_type
        self.inls2d_thres = config.inls2d_thres
        self.normalized_thres = config.normalized_thres
        self.npts = config.npts
        self.outlier_rate = config.outlier_rate

        # Load dataset configs
        self.split = split
        self.data_root = config.data_root
        self.dataset = config.dataset

        # Load all rooms for training
        self.scenes = sorted(glob(os.path.join(self.data_root, '*')))
        self.rooms = []
        for scene in self.scenes:
            self.rooms += sorted(glob(os.path.join(scene, '*')))
        
        train_size = int(len(self.rooms) * 0.9)
        if split == 'train':
            self.rooms = self.rooms[:train_size]
        else:
            self.rooms = self.rooms[train_size:]
        
        self.view_pairs = []
        for room in self.rooms:
            self.view_pairs += sorted(glob(os.path.join(room, '*.npz')))


    def _construct_data(
        self,
        view_pair: Dict
    ) -> Dict[str, Any]:

        orig_R = view_pair['R']
        orig_T = view_pair['T']
        orig_sphere = view_pair['kpts_sphere']
        orig_ij = view_pair['kpts_ij']
        orig_3d = view_pair['kpts_3d']

        new_R = view_pair['R_pair']
        new_T = view_pair['T_pair']
        new_sphere = view_pair['kpts_sphere_pair']
        new_ij = view_pair['kpts_ij_pair']
        new_3d = view_pair['kpts_3d_pair']

        # Point subsampling might be needed after merging
        if len(orig_sphere) > self.npts[1]:
            orig_rids = np.random.choice(len(orig_sphere), self.npts[1], replace=False)
            orig_sphere = orig_sphere[orig_rids]
            orig_ij = orig_ij[orig_rids]
            orig_3d = orig_3d[orig_rids]
        elif len(orig_sphere) < self.npts[0]:
            orig_sphere = np.empty([0, 3])
            orig_ij = np.empty([0, 2])
            orig_3d = np.empty([0, 3])
        if len(new_sphere) > self.npts[1]:
            new_rids = np.random.choice(len(new_sphere), self.npts[1], replace=False)
            new_sphere = new_sphere[new_rids]
            new_ij = new_ij[new_rids]
            new_3d = new_3d[new_rids]
        elif len(new_sphere) < self.npts[0]:
            new_sphere = np.empty([0, 3])
            new_ij = np.empty([0, 2])
            new_3d = np.empty([0, 3])

        # Compute pesudo ground truth for 2d 3d matching
        i2ds, i3ds, o2ds, o3ds = compute_gt_2d3d_match(
            orig_sphere,
            new_3d,
            orig_R,
            orig_T,
            inls_thres=self.inls2d_thres
        )

        if self.outlier_rate == [0, 1]:
            # Generate assignment mask
            n2d, n3d = len(orig_sphere), len(new_3d)
            matches_bin = np.zeros((n3d + 1, n2d + 1), dtype=bool)
            matches_bin[i3ds, i2ds] = True
            matches_bin[o3ds, -1] = True
            matches_bin[-1, o2ds] = True
        else:
            # Control outliers for training and ablation
            pts2d_ids, pts3d_ids, matches = subsample_points_indices(
                i2ds, i3ds, o2ds, o3ds, self.outlier_rate, self.npts
            )
            orig_sphere = orig_sphere[pts2d_ids]
            orig_3d = orig_3d[pts2d_ids]
            orig_ij = orig_ij[pts2d_ids]

            new_sphere = new_sphere[pts3d_ids]
            new_3d = new_3d[pts3d_ids]
            new_ij = new_ij[pts3d_ids]

            # Generate assignment mask
            matches_bin = generate_assignment_mask(matches, len(new_3d))

        # Construct data
        data = dict(
            orig_sphere=orig_sphere.astype(np.float32),
            orig_3d=orig_3d.astype(np.float32),
            orig_ij=orig_ij.astype(np.float32),
            new_sphere=new_sphere.astype(np.float32),
            new_3d=new_3d.astype(np.float32),
            new_ij=new_ij.astype(np.float32),
            matches_bin=matches_bin,
            orig_R=orig_R.astype(np.float32),
            orig_T=orig_T.astype(np.float32),
            new_R=new_R.astype(np.float32),
            new_T=new_T.astype(np.float32)
        )

        return data

    def __getitem__(self, index: int) -> Dict[str, Any]:
        view_pair = self.view_pairs[index]
        view_pair = np.load(view_pair)

        # Generate kpts and labels for 2d-3d matching
        data = self._construct_data(view_pair)
        return data

    def __len__(self) -> int:
        return len(self.view_pairs)

    def __repr__(self) -> str:
        fmt_str = f"\nDataset:{self.dataset} split={self.split} "
        fmt_str += f"Data processed dir: {self.data_root}\n"
        fmt_str += f"Settings=(\n"
        fmt_str += f"  feat_type={self.feat_type}, orate={self.outlier_rate}, npt={self.npts},\n"
        fmt_str += f"  inls_thres={self.inls2d_thres} normalized_thres={self.normalized_thres}\n"
        fmt_str += f")\n"
        return fmt_str
