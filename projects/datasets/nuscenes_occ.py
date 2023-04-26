from os import path as osp
import copy
import numpy as np
from typing import Callable, List, Union
import random

import torch

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

    def parse_ann_info(self, info: dict) -> dict:
        ann_info = super().parse_ann_info(info)
        occ_gt_path = osp.join(self.data_root,info['occ_gt_path'])
        occ_gt_label = np.load(occ_gt_path)
        ann_info['occ_semantics'] = occ_gt_label['semantics']
        ann_info['mask_lidar'] = occ_gt_label['mask_lidar']
        ann_info['mask_camera'] = occ_gt_label['mask_camera']
        return ann_info

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

    def prepare_data(self, index: int) -> Union[dict, None]:
        """Data preparation for both training and testing stage.

        Called by `__getitem__`  of dataset.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict or None: Data dict of the corresponding index.
        """
        ori_input_dict = self.get_data_info(index)

        # deepcopy here to avoid inplace modification in pipeline.
        input_dict = copy.deepcopy(ori_input_dict)

        # box_type_3d (str): 3D box type.
        input_dict['box_type_3d'] = self.box_type_3d
        # box_mode_3d (str): 3D box mode.
        input_dict['box_mode_3d'] = self.box_mode_3d

        # pre-pipline return None to random another in `__getitem__`
        if not self.test_mode and self.filter_empty_gt:
            if len(input_dict['ann_info']['gt_labels_3d']) == 0:
                return None

        if not self.test_mode:
            queue = []
            index_list = list(range(index - self.queue_length, index))
            random.shuffle(index_list)
            index_list = sorted(index_list[1:])
            index_list.append(index)
            for i in index_list:
                i = max(0, i)
                ori_input_dict = self.get_data_info(i)
                if ori_input_dict is None:
                    return None
                input_dict = copy.deepcopy(ori_input_dict)
                example = self.pipeline(input_dict)
                queue.append(example)
            queue = self.union2one(queue)
            return queue

        example = self.pipeline(input_dict)

        return example

    def union2one(self, queue):
        imgs_list = [each['inputs']['img'] for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['data_samples']
            if metas_map[i].scene_token != prev_scene_token:
                metas_map[i].prev_bev_exists = False
                prev_scene_token = metas_map[i].scene_token
                prev_pos = copy.deepcopy(metas_map[i].can_bus[:3])
                prev_angle = copy.deepcopy(metas_map[i].can_bus[-1])
                metas_map[i].can_bus[:3] = 0
                metas_map[i].can_bus[-1] = 0
            else:
                metas_map[i].prev_bev_exists = True
                tmp_pos = copy.deepcopy(metas_map[i].can_bus[:3])
                tmp_angle = copy.deepcopy(metas_map[i].can_bus[-1])
                metas_map[i].can_bus[:3] -= prev_pos
                metas_map[i].can_bus[-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
        queue[-1]['inputs']['img'] = torch.stack(imgs_list)
        queue[-1]['data_samples'] = metas_map
        queue = queue[-1]
        return queue


