import os
import numpy as np

from mmdet3d.registry import TRANSFORMS
from mmcv.transforms.base import BaseTransform


@TRANSFORMS.register_module()
class LoadOccGTFromFile(BaseTransform):
    def __init__(self,
                 data_root) -> None:
        self.data_root = data_root

    def transform(self, results: dict) -> dict:
        occ_gt_path = results['occ_gt_path']
        occ_gt_path = os.path.join(self.data_root, occ_gt_path)

        occ_labels = np.load(occ_gt_path)
        semantics = occ_labels['semantics']
        mask_lidar = occ_labels['mask_lidar']
        mask_camera = occ_labels['mask_camera']

        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(data_root={self.data_root}, '
        return repr_str
