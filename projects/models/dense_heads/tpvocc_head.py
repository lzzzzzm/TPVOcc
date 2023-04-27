import torch
import torch.nn as nn

from mmengine.model import BaseModule

from mmdet3d.registry import MODELS

from mmdet.registry import MODELS as MMDETMODELS

@MODELS.register_module()
class TPVOccHead(BaseModule):
    def __init__(self,
                 *args,
                 transformer=None,
                 pc_range=[-40, -40, -1.0, 40, 40, 5.4],
                 bev_h=30,
                 bev_w=30,
                 num_classes=18,
                 embed_dims=256,
                 positional_encoding=dict(
                     type='mmdet.LearnedPositionalEncoding',
                     num_feats=128,
                     row_num_embed=200,
                     col_num_embed=200
                 ),
                 loss_occ=dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0
                 ),
                 loss_lovasz=None,
                 use_mask=False,
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        super(TPVOccHead, self).__init__()
        self.fp16_enabled = False

        self.use_mask = use_mask
        self.num_classes = num_classes
        self.transformer = MODELS.build(transformer)
        self.positional_encoding = MMDETMODELS.build(positional_encoding)
        self.loss_occ = MODELS.build(loss_occ)
        if loss_lovasz is not None:
            self.loss_lovasz = MODELS.build(loss_lovasz)
        else:
            self.loss_lovasz = None
        self.bev_embedding = nn.Embedding(
            self.bev_h * self.bev_w, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()

    def get_occ(self, preds_dicts):
        """Generate bboxes from bbox head predictions.
        Args:
            predss : occ results.
        Returns:
            list[dict]: occ score: [bs, bev_h, bev_w, height]
        """
        occ_out=preds_dicts['occ']
        occ_score=occ_out.softmax(-1)
        occ_score=occ_score.argmax(-1)
        return occ_score

    def loss(self,
             occ_semantics,
             mask_camera,
             preds_dicts,
             ):

        occ = preds_dicts['occ']
        assert occ_semantics.min() >= 0 and occ_semantics.max() <= 17
        loss_dict = self.loss_single(occ_semantics, mask_camera, occ)
        return loss_dict

    def loss_single(self,
                    voxel_semantics,
                    mask_camera,
                    preds):
        loss_dict = dict()
        voxel_semantics=voxel_semantics.long()
        if self.use_mask:
            voxel_semantics=voxel_semantics.reshape(-1)
            preds=preds.reshape(-1,self.num_classes)
            mask_camera=mask_camera.reshape(-1)
            num_total_samples=mask_camera.sum()
            loss_occ=self.loss_occ(preds,voxel_semantics,mask_camera, avg_factor=num_total_samples)
            loss_dict['loss_occ'] = loss_occ
        else:
            if self.loss_lovasz is not None:
                loss_lovasz = self.loss_lovasz(preds, voxel_semantics)
                loss_dict['loss_lovasz'] = loss_lovasz
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics)
            loss_dict['loss_occ'] = loss_occ

        return loss_dict

    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = None
        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev
            )
        bev_embed, occ_outs = outputs
        outs = {
            'bev_embed': bev_embed,
            'occ': occ_outs,
        }
        return outs


