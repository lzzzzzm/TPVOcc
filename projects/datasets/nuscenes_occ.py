from os import path as osp
import numpy as np
from typing import Callable, List, Union

from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.utils.geometry_utils import transform_matrix

from mmdet3d.datasets import NuScenesDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class NuScenesOcc(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self,
                 queue_length=4,
                 tpv_size=(200, 200, 16),
                 overlap_test=False,
                 eval_fscore=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_fscore = eval_fscore
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.tpv_size = tpv_size

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        data_info = super().parse_data_info(info)
        # ego2lidar
        lidar2ego_rotation = info['lidar_points']['lidar2ego_rotation']
        lidar2ego_translation = info['lidar_points']['lidar2ego_translation']
        ego2lidar = transform_matrix(translation=lidar2ego_translation, rotation=Quaternion(lidar2ego_rotation),
                                     inverse=True)
        data_info['ego2lidar'] = ego2lidar
        # scene_token
        data_info['scene_token'] = info['scene_token']
        # lidar2img
        lidar2cam_rts = []
        for cam_type, cam_info in info['images'].items():
            lidar2cam_rts.append(cam_info['lidar2img'])
        data_info['lidar2img'] = lidar2cam_rts
        # can bus
        rotation = Quaternion(data_info['ego2global_rotation'])
        translation = data_info['ego2global_translation']
        can_bus = data_info['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle
        return data_info