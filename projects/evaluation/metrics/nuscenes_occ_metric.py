import numpy as np
import os
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Optional, Sequence, Tuple, Union

from mmengine import Config, load
from mmengine.logging import MMLogger
from nuscenes.eval.detection.config import config_factory

from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS


@METRICS.register_module()
class NuScenesOccMetric(BaseMetric):
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 modality: dict = dict(use_camera=False, use_lidar=True),
                 prefix: Optional[str] = None,
                 format_only: bool = False,
                 jsonfile_prefix: Optional[str] = None,
                 use_image_mask=True,
                 use_lidar_mask=False,
                 num_classes=18,
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'NuScenes Occ metric'
        super(NuScenesOccMetric, self).__init__(
            collect_device=collect_device, prefix=prefix)
        self.class_names = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                            'driveable_surface', 'other_flat', 'sidewalk',
                            'terrain', 'manmade', 'vegetation', 'free']
        if modality is None:
            modality = dict(
                use_camera=True,
                use_lidar=False,
            )
        self.ann_file = ann_file
        self.data_root = data_root
        self.modality = modality
        self.format_only = format_only
        if self.format_only:
            assert jsonfile_prefix is not None, 'jsonfile_prefix must be not '
            'None when format_only is True, otherwise the result files will '
            'be saved to a temp directory which will be cleanup at the end.'

        self.jsonfile_prefix = jsonfile_prefix
        self.backend_args = backend_args

        self.metrics = metric if isinstance(metric, list) else [metric]
        self.use_image_mask = use_image_mask
        self.use_lidar_mask = use_lidar_mask
        self.num_classes = num_classes
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cm = np.zeros(shape=(self.num_classes, self.num_classes))

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample['pred_instances_3d']
            pred_2d = data_sample['pred_instances']
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_3d
            for attr_name in pred_2d:
                pred_2d[attr_name] = pred_2d[attr_name].to('cpu')
            result['pred_instances'] = pred_2d
            sample_idx = data_sample['sample_idx']
            result['sample_idx'] = sample_idx
            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:

        logger: MMLogger = MMLogger.get_current_instance()

        self.version = self.dataset_meta['version']
        # load annotations
        self.data_infos = load(
            self.ann_file, backend_args=self.backend_args)['data_list']

        # TODO: Only support batch size = 1
        for i in range(len(self.data_infos)):
            occ_gt_path = self.data_infos[i]['occ_gt_path']
            occ_gt_path = os.path.join(self.data_root, occ_gt_path)
            occ_gt = np.load(occ_gt_path)
            label = occ_gt['semantics']
            mask_lidar = np.array(occ_gt['mask_lidar'], dtype=np.bool)
            mask_camera = np.array(occ_gt['mask_camera'], dtype=np.bool)
            # pred
            pred = self.results[i]['pred_instances_3d']['labels_3d'].cpu().numpy().squeeze()
            if self.use_image_mask:
                label = label[mask_camera]
                pred = pred[mask_camera]
            if self.use_lidar_mask:
                label = label[mask_lidar]
                pred = pred[mask_lidar]
            _, _hist = self.compute_mIoU(pred, label, self.num_classes)
            self.hist += _hist
        metric_dict = self.count_miou(logger=logger)
        return metric_dict

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        return round(np.nanmean(mIoUs) * 100, 2), hist

    def per_class_iu(self, hist):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-6)

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape
        cm = confusion_matrix(gt, pred, labels=np.arange(0, self.num_classes))
        self.cm += cm
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def count_miou(self, logger):
        result_dict = {}
        mIoU = self.per_class_iu(self.hist)
        for ind_class in range(self.num_classes - 1):
            logger.info(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU[ind_class] * 100, 2)))
        logger.info(f'===> mIoU of {len(self.results)} samples: ' + str(round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2)))
        result_dict['mIoU'] = round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2)
        return result_dict
