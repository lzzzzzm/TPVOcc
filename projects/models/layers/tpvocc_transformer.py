import numpy as np

import torch
import torch.nn as nn
from torchvision.transforms.functional import rotate

from mmengine.model import BaseModule, normal_init, xavier_init

from mmcv.cnn import ConvModule

from mmdet3d.registry import MODELS

from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D

@MODELS.register_module()
class TPVOccTransformer(BaseModule):
    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 pillar_h=16,
                 num_classes=18,
                 out_dim=32,
                 embed_dims=256,
                 rotate_center=[100, 100],
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 use_3d=True,
                 use_conv=False,
                 encoder=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='BN', ),
                 norm_cfg_3d=dict(type='BN3d', ),
                 **kwargs
                 ):

        super(TPVOccTransformer, self).__init__(**kwargs)
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.pillar_h = pillar_h
        self.num_classes = num_classes
        self.out_dim = out_dim
        self.embed_dims = embed_dims
        self.rotate_center = rotate_center
        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_can_bus = use_can_bus
        self.use_cams_embeds = use_cams_embeds
        self.encoder = MODELS.build(encoder)
        self.use_3d = use_3d
        self.use_conv = use_conv
        if not use_3d:
            if use_conv:
                use_bias = norm_cfg is None
                self.decoder = nn.Sequential(
                    ConvModule(
                        self.embed_dims,
                        self.embed_dims,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=use_bias,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg),
                    ConvModule(
                        self.embed_dims,
                        self.embed_dims * 2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=use_bias,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg), )

            else:
                self.decoder = nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims * 2),
                    nn.Softplus(),
                    nn.Linear(self.embed_dims * 2, self.embed_dims * 2),
                )
        else:
            use_bias_3d = norm_cfg_3d is None

            self.middle_dims = self.embed_dims // pillar_h
            self.decoder = nn.Sequential(
                ConvModule(
                    self.middle_dims,
                    self.out_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias_3d,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=norm_cfg_3d,
                    act_cfg=act_cfg),
                ConvModule(
                    self.out_dim,
                    self.out_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias_3d,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=norm_cfg_3d,
                    act_cfg=act_cfg),
            )
        self.predicter = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim * 2),
            nn.Softplus(),
            nn.Linear(self.out_dim * 2, num_classes),
        )
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        # self.reference_points = nn.Linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_init(self.level_embeds)
        normal_init(self.cams_embeds)
        # xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """

        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion

        delta_x = np.array([each['can_bus'][0]
                           for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]
                           for each in kwargs['img_metas']])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        xy_shitft = np.array([shift_x, shift_y])
        shift = bev_queries.new_tensor(xy_shitft).permute(1, 0)  # xy, bs -> bs, xy

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)

            elif len(prev_bev.shape) == 4:
                prev_bev = prev_bev.view(bs,-1,bev_h * bev_w).permute(2, 0, 1)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        each_can_bus = [np.array(each['can_bus']) for each in kwargs['img_metas']]
        can_bus = bev_queries.new_tensor(np.array(each_can_bus))  # [:, :]
        # TODO: only support batch size=1
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs
        )

        return bev_embed

    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                prev_bev=None,
                **kwargs):
        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs = mlvl_feats[0].size(0)
        bev_embed = bev_embed.permute(0, 2, 1).view(bs, -1, bev_h, bev_w)
        if self.use_3d:
            outputs = self.decoder(bev_embed.view(bs, -1, self.pillar_h, bev_h, bev_w))
            outputs = outputs.permute(0, 4, 3, 2, 1)
        elif self.use_conv:
            outputs = self.decoder(bev_embed)
            outputs = outputs.view(bs, -1, self.pillar_h, bev_h, bev_w).permute(0, 3, 4, 2, 1)
        else:
            outputs = self.decoder(bev_embed.permute(0, 2, 3, 1))
            outputs = outputs.view(bs, bev_h, bev_w, self.pillar_h, self.out_dim)
        outputs = self.predicter(outputs)
        return bev_embed, outputs

