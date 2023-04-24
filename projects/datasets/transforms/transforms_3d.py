import cv2

from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS

@TRANSFORMS.register_module()
class ResizeMultiViewImage(BaseTransform):
    def __init__(self, size) -> None:
        self.size = size

    def transform(self, input_dict: dict) -> dict:
        imgs = input_dict['img']
        resize_imgs = [cv2.resize(img, self.size) for img in imgs]
        input_dict['img'] = resize_imgs
        return input_dict
