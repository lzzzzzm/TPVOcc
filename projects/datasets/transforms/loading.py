import os
import numpy as np

from mmdet3d.registry import TRANSFORMS
from mmcv.transforms.base import BaseTransform


@TRANSFORMS.register_module()
class LoadOccAnnotations3D(BaseTransform):
    def __init__(self,
                 with_occ_semantics=True,
                 with_camera_mask=True,
                 with_lidar_mask=False,
                 **kwargs) -> None:
        super().__init__()
        self.with_occ_semantics = with_occ_semantics
        self.with_camera_mask = with_camera_mask
        self.with_lidar_mask = with_lidar_mask

    def _load_occ_semantic(self, results: dict):
        results['occ_semantics'] = results['ann_info']['occ_semantics']
        return results

    def _load_camera_mask(self, results: dict):
        results['mask_camera'] = results['ann_info']['mask_camera']
        return results

    def _load_lidar_mask(self, results: dict):
        results['mask_lidar'] = results['ann_info']['mask_lidar']
        return results


    def transform(self, results: dict) -> dict:
        if self.with_occ_semantics:
            results = self._load_occ_semantic(results)
        if self.with_camera_mask:
            results = self._load_camera_mask(results)
        if self.with_lidar_mask:
            results = self._load_lidar_mask(results)

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_occ_semantic={self.with_occ_semantic}, '
        repr_str += f'{indent_str}with_camera_mask={self.with_camera_mask}, '
        repr_str += f'{indent_str}with_lidar_mask={self.with_lidar_mask}, '

        return repr_str
