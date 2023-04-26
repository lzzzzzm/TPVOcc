from typing import Dict, List, Optional
import copy

import torch
from torch import Tensor

from mmengine.structures import InstanceData
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.structures import Det3DDataSample
from mmdet3d.registry import MODELS

from .grid_mask import GridMask
@MODELS.register_module()
class TPVOcc(MVXTwoStageDetector):
    def __init__(self,
                 data_preprocessor=None,
                 use_grid_mask=False,
                 img_backbone=None,
                 img_neck=None,
                 pts_bbox_head=None,
                 video_test_mode=False):
        
        super().__init__(data_preprocessor=data_preprocessor,
                         img_backbone=img_backbone,
                         img_neck=img_neck,
                         pts_bbox_head=pts_bbox_head)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }


    def extract_img_feat(self, img: Tensor,
                         len_queue = None) -> List[Tensor]:
        """Extract features from images.

        Args:
            img (tensor): Batched multi-view image tensor with
                shape (B, N, C, H, W).
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.

        Returns:
             list[tensor]: multi-level image features.
        """

        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)  # mask out some grids
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

         Args:
             batch_inputs_dict (dict): The model input dict which include
                 'points' keys.

                 - points (list[torch.Tensor]): Point cloud of each sample.
             batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                 Samples. It usually includes information such as
                 `gt_instance_3d`.

         Returns:
             list[:obj:`Det3DDataSample`]: Detection results of the
             input sample. Each Det3DDataSample usually contain
             'pred_instances_3d'. And the ``pred_instances_3d`` usually
             contains following keys.

             - scores_3d (Tensor): Classification scores, has a shape (h, w, z)
             - labels_3d (Tensor): Labels of occ, has a shape (h, w, z)
         """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        if batch_input_metas[0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        self.prev_frame_info['scene_token'] = batch_input_metas[0]['scene_token']
        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        tmp_pos = copy.deepcopy(batch_input_metas[0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(batch_input_metas[0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            batch_input_metas[0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            batch_input_metas[0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            batch_input_metas[0]['can_bus'][-1] = 0
            batch_input_metas[0]['can_bus'][:3] = 0

        new_prev_bev, occ_results = self.simple_test(batch_inputs_dict, batch_input_metas,
                                                     prev_bev=self.prev_frame_info['prev_bev'])

        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev

        # ret
        ret_list = []
        results_list_3d = InstanceData()
        results_list_3d.labels_3d = occ_results
        ret_list.append(results_list_3d)
        detsamples = self.add_pred_to_datasample(batch_data_samples,
                                                 ret_list)
        return detsamples

    def simple_test_pts(self, img_feats, img_metas, prev_bev=None):
        """Test function"""
        outs = self.pts_bbox_head(img_feats, img_metas, prev_bev=prev_bev)
        occ = self.pts_bbox_head.get_occ(outs)
        return outs['bev_embed'], occ

    def simple_test(self, batch_inputs_dict, batch_input_metas, prev_bev=None):
        """Test function without augmentaiton."""
        imgs = batch_inputs_dict.get('imgs', None)
        img_feats = self.extract_img_feat(imgs)
        new_prev_bev, occ = self.simple_test_pts(
            img_feats, batch_input_metas, prev_bev)
        return new_prev_bev, occ

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        self.eval()
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_img_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    def forward_pts_train(self,
                          img_feats,
                          occ_semantics,
                          img_metas,
                          mask_camera=None,
                          prev_bev=None):
        outs = self.pts_bbox_head(img_feats, img_metas, prev_bev=prev_bev)
        loss_inputs = [occ_semantics, mask_camera, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def loss(self, batch_inputs_dict: Dict[List, torch.Tensor],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        # process img metas
        # TODO : only support batch size = 1
        batch_input_metas = {}
        for key in batch_data_samples:
            index_batch_meta_info = batch_data_samples[key][0].metainfo
            # update meta info
            index_batch_meta_info['scene_token'] = batch_data_samples[key][0].scene_token
            index_batch_meta_info['prev_bev_exists'] = batch_data_samples[key][0].prev_bev_exists
            batch_input_metas[key] = index_batch_meta_info
        img_metas = [batch_input_metas]
        imgs = batch_inputs_dict.get('imgs', None)

        len_queue = imgs.size(1)
        prev_img = imgs[:, :-1, ...]
        imgs = imgs[:, -1, ...]
        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue - 1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None

        img_feats = self.extract_img_feat(imgs)
        losses = dict()
        occ_semantics = batch_data_samples[len(batch_data_samples)-1][0].gt_instances_3d.occ_semantics
        losses_pts = self.forward_pts_train(img_feats,
                                            occ_semantics,
                                            img_metas,
                                            prev_bev=prev_bev)
        losses.update(losses_pts)
        return losses
