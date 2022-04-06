import os
import time
import random
import numpy as np

from lib.config import CONF
from utils.pc_utils import random_sampling, rotx, roty, rotz, pad_samples
from utils.box_util import get_3d_box, get_3d_box_batch
from data.scannet.model_util_scannet import ScannetDatasetConfig, rotate_aligned_boxes_along_axis
from lib.reference_dataset import ReferenceDataset

# data setting
DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 64
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])


class Dataset(ReferenceDataset):
    def __init__(self, scanrefer, scanrefer_all_scene, name, split="train", num_points=40000,
                 use_height=False, use_color=False, use_normal=False, augment=False):

        super().__init__()
        self.scanrefer = scanrefer
        self.scanrefer_all_scene = scanrefer_all_scene  # all scene_ids in scanrefer
        self.split = split
        self.name = name
        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.use_normal = use_normal
        self.augment = augment

        # load data
        self._load_data(name)
        self.multiview_data = {}
        self.gt_feature_data = {}

        # fliter
        self.scene_objects = self._get_scene_objects(self.scanrefer)
        self.data_2d = np.load(CONF.PATH.DATA_2D, allow_pickle=True)
        self.data_2d = self.data_2d['arr_0'].item()

    def __len__(self):
        return len(self.scanrefer)

    def __getitem__(self, idx):
        start = time.time()
        scene_id = self.scanrefer[idx]["scene_id"]
        object_id = int(self.scanrefer[idx]["object_id"])
        object_name = " ".join(self.scanrefer[idx]["object_name"].split("_"))
        ann_id = self.scanrefer[idx]["ann_id"]

        annotated = 1

        object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17

        # get language features
        # lang_feat = self.lang[scene_id][str(object_id)][ann_id]
        lang_len = len(self.scanrefer[idx]["token"]) + 2
        lang_len = lang_len if lang_len <= CONF.TRAIN.MAX_DES_LEN + 2 else CONF.TRAIN.MAX_DES_LEN + 2

        # lang_ids = self.lang_ids[scene_id][str(object_id)][ann_id]

        unique_multiple_flag = self.unique_multiple_lookup[scene_id][str(object_id)][ann_id]

        # get pc
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]
        instance_labels = self.scene_data[scene_id]["instance_labels"]
        semantic_labels = self.scene_data[scene_id]["semantic_labels"]
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]

        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
            pcl_color = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, 0:6]
            point_cloud[:, 3:6] = (point_cloud[:, 3:6] - MEAN_COLOR_RGB) / 256.0
            pcl_color = point_cloud[:, 3:6]

        if self.use_normal:
            normals = mesh_vertices[:, 6:9]
            point_cloud = np.concatenate([point_cloud, normals], 1)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]

        # ------------------------------- LABELS ------------------------------
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))

        ref_box_label = np.zeros(MAX_NUM_OBJ)  # bbox label for reference target
        ref_center_label = np.zeros(3)  # bbox center for reference target
        ref_heading_class_label = 0
        ref_heading_residual_label = 0
        ref_size_class_label = 0
        ref_size_residual_label = np.zeros(3)  # bbox size residual for reference target
        ref_box_corner_label = np.zeros((8, 3))

        num_bbox = 1
        point_votes = np.zeros([self.num_points, 3])
        point_votes_mask = np.zeros(self.num_points)

        num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < MAX_NUM_OBJ else MAX_NUM_OBJ
        target_bboxes_mask[0:num_bbox] = 1
        target_bboxes[0:num_bbox, :] = instance_bboxes[:MAX_NUM_OBJ, 0:6]

        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

                # Rotation along X-axis
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            rot_mat = rotx(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "x")

            # Rotation along Y-axis
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            rot_mat = roty(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "y")

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            rot_mat = rotz(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "z")

            # Translation
            point_cloud, target_bboxes = self._translate(point_cloud, target_bboxes)

        # compute votes *AFTER* augmentation
        # generate votes
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label.
        for i_instance in np.unique(instance_labels):
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            # find the semantic label
            if semantic_labels[ind[0]] in DC.nyu40ids:
                x = point_cloud[ind, :3]
                center = 0.5 * (x.min(0) + x.max(0))
                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
        point_votes = np.tile(point_votes, (1, 3))  # make 3 votes identical

        class_ind = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:num_bbox, -2]]
        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:num_bbox] = class_ind
        size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - DC.mean_size_arr[class_ind, :]

        # construct the reference target label for each bbox
        if object_id != -1:
            for i, gt_id in enumerate(instance_bboxes[:num_bbox, -1]):
                if gt_id == object_id:
                    ref_box_label[i] = 1
                    ref_center_label = target_bboxes[i, 0:3]
                    ref_heading_class_label = angle_classes[i]
                    ref_heading_residual_label = angle_residuals[i]
                    ref_size_class_label = size_classes[i]
                    ref_size_residual_label = size_residuals[i]

                    # construct ground truth box corner coordinates
                    ref_obb = DC.param2obb(ref_center_label, ref_heading_class_label, ref_heading_residual_label,
                                           ref_size_class_label, ref_size_residual_label)
                    ref_box_corner_label = get_3d_box(ref_obb[3:6], ref_obb[6], ref_obb[0:3])

        # construct all GT bbox corners
        all_obb = DC.param2obb_batch(target_bboxes[:num_bbox, 0:3], angle_classes[:num_bbox].astype(np.int64),
                                     angle_residuals[:num_bbox],
                                     size_classes[:num_bbox].astype(np.int64), size_residuals[:num_bbox])
        all_box_corner_label = get_3d_box_batch(all_obb[:, 3:6], all_obb[:, 6], all_obb[:, 0:3])

        # store
        gt_box_corner_label = np.zeros((MAX_NUM_OBJ, 8, 3))
        gt_box_masks = np.zeros((MAX_NUM_OBJ,))
        gt_box_object_ids = np.zeros((MAX_NUM_OBJ,))
        gt_box_obb = np.zeros((MAX_NUM_OBJ, 7))

        gt_box_corner_label[:num_bbox] = all_box_corner_label
        gt_box_masks[:num_bbox] = 1
        gt_box_object_ids[:num_bbox] = instance_bboxes[:num_bbox, -1]
        gt_box_obb[:num_bbox] = all_obb

        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_object_ids = np.zeros((MAX_NUM_OBJ,))  # object ids of all objects
        try:
            target_bboxes_semcls[0:num_bbox] = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:, -2][0:num_bbox]]
            target_object_ids[0:num_bbox] = instance_bboxes[:, -1][0:num_bbox]
        except KeyError:
            pass

        # get bbox
        bbox_mask = np.zeros((MAX_NUM_OBJ,))
        bbox_mask[:num_bbox] = 1

        # load 2d features
        bbox_2d, feat_2d, cls_2d, attrs = self.load_2d_features(scene_id, instance_bboxes[:num_bbox, -1])

        # load instance point cloud
        instance_points = self.load_instance_pointcloud(point_cloud, instance_bboxes[:num_bbox, -1], instance_labels)

        bbox_idx = 0
        for i in range(len(gt_box_object_ids)):
            if gt_box_object_ids[i] == object_id:
                bbox_idx = i
                break

        # object rotations
        scene_object_rotations = np.zeros((MAX_NUM_OBJ, 3, 3))
        scene_object_rotation_masks = np.zeros((MAX_NUM_OBJ,))  # NOTE this is not object mask!!!

        data_dict = {}
        # pc
        data_dict["point_clouds"] = point_cloud.astype(np.float32)  # point cloud data including features
        data_dict["pcl_color"] = pcl_color

        # basic info
        data_dict["dataset_idx"] = np.array(idx).astype(np.int64)
        # data_dict["ann_id"] = ann_id  # np.array(int(ann_id)).astype(np.int64)
        data_dict["object_id"] = np.array(int(object_id)).astype(np.int64)
        data_dict["object_cat"] = np.array(object_cat).astype(np.int64)
        data_dict["annotated"] = np.array(annotated).astype(np.int64)

        # language data
        # data_dict["lang_feat"] = lang_feat.astype(np.float32)  # language feature vectors
        data_dict["lang_len"] = np.array(lang_len).astype(np.int64)  # length of each description
        data_dict["lang_ids"] = np.array(self.lang_ids[scene_id][str(object_id)][ann_id]).astype(np.int64)

        # GT bbox data
        data_dict["bbox_mask"] = bbox_mask.astype(np.int64)  # mask indicating the valid objects
        data_dict["bbox_idx"] = bbox_idx  # idx for the target object

        # object detection labels
        data_dict["center_label"] = target_bboxes.astype(np.float32)[:, 0:3]  # (MAX_NUM_OBJ, 3) for GT box center XYZ
        data_dict["heading_class_label"] = angle_classes.astype(
            np.int64)  # (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
        data_dict["heading_residual_label"] = angle_residuals.astype(np.float32)  # (MAX_NUM_OBJ,)
        data_dict["size_class_label"] = size_classes.astype(
            np.int64)  # (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
        data_dict["size_residual_label"] = size_residuals.astype(np.float32)  # (MAX_NUM_OBJ, 3)
        data_dict["num_bbox"] = np.array(num_bbox).astype(np.int64)
        data_dict["sem_cls_label"] = target_bboxes_semcls.astype(np.int64)  # (MAX_NUM_OBJ,) semantic class index
        data_dict["scene_object_ids"] = target_object_ids.astype(np.int64)  # (MAX_NUM_OBJ,) object ids of all objects
        data_dict["box_label_mask"] = target_bboxes_mask.astype(
            np.float32)  # (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
        data_dict["vote_label"] = point_votes.astype(np.float32)
        data_dict["vote_label_mask"] = point_votes_mask.astype(np.int64)

        # localization labels
        data_dict["ref_box_label"] = ref_box_label.astype(np.int64)  # 0/1 reference labels for each object bbox
        data_dict["ref_center_label"] = ref_center_label.astype(np.float32)
        data_dict["ref_heading_class_label"] = np.array(int(ref_heading_class_label)).astype(np.int64)
        data_dict["ref_heading_residual_label"] = np.array(int(ref_heading_residual_label)).astype(np.int64)
        data_dict["ref_size_class_label"] = np.array(int(ref_size_class_label)).astype(np.int64)
        data_dict["ref_size_residual_label"] = ref_size_residual_label.astype(np.float32)
        data_dict["ref_box_corner_label"] = ref_box_corner_label.astype(
            np.float64)  # target box corners NOTE type must be double
        data_dict["unique_multiple"] = np.array(unique_multiple_flag).astype(np.int64)

        # ground truth data

        data_dict["gt_box_corner_label"] = gt_box_corner_label.astype(
            np.float64)  # all GT box corners NOTE type must be double
        data_dict["gt_box_masks"] = gt_box_masks.astype(np.int64)  # valid bbox masks
        data_dict["gt_box_object_ids"] = gt_box_object_ids.astype(np.int64)  # valid bbox object ids

        data_dict["bbox_object_ids"] = data_dict["gt_box_object_ids"]

        # rotation data
        data_dict["scene_object_rotations"] = scene_object_rotations.astype(np.float32)  # (MAX_NUM_OBJ, 3, 3)
        data_dict["scene_object_rotation_masks"] = scene_object_rotation_masks.astype(np.int64)  # (MAX_NUM_OBJ)

        data_dict['bbox_obb'] = gt_box_obb.astype(np.float32)
        data_dict['bbox_2d'] = bbox_2d
        data_dict['feat_2d'] = feat_2d
        data_dict['objects'] = instance_points.astype(np.float32)

        # misc
        data_dict["load_time"] = time.time() - start

        return data_dict

    def load_2d_features(self, scan_id, gt_object_ids):
        data_2d = self.data_2d[scan_id]

        bbox = np.zeros((MAX_NUM_OBJ, 4), dtype=np.float32)
        feat = np.zeros((MAX_NUM_OBJ, 2048), dtype=np.float32)
        cls_2d = np.zeros(MAX_NUM_OBJ, dtype=np.long)
        attrs = np.zeros(MAX_NUM_OBJ, dtype=np.long)

        for i, gt_id in enumerate(gt_object_ids):
            object_id = int(gt_id) + 1
            if object_id in data_2d:
                item_2d = data_2d[object_id]
                idx = random.randint(0, len(item_2d) - 1)

                bbox[i] = item_2d[idx]['bbox']
                feat[i] = item_2d[idx]['feat']
                cls_2d[i] = item_2d[idx]['objects_id']
                attrs[i] = item_2d[idx]['attrs_id']
        return bbox, feat, cls_2d, attrs

    def load_instance_pointcloud(self, point_cloud, gt_object_ids, instance_labels):
        # instanc_points = np.zeros((MAX_NUM_OBJ, 1024, 6))
        instance_points = []
        for i, gt_id in enumerate(gt_object_ids):
            object_id = int(gt_id) + 1
            ind_mask = instance_labels == object_id
            if ind_mask.sum() == 0:
                continue
            pc_in_box = random_sampling(point_cloud[ind_mask], 1024)
            instance_points.append(pc_in_box)

        instance_points = np.asarray(instance_points)
        instance_points = pad_samples(instance_points, MAX_NUM_OBJ)

        return instance_points

    def _get_scene_objects(self, data):
        scene_objects = {}
        for d in data:
            scene_id = d["scene_id"]
            object_id = d["object_id"]

            if scene_id not in scene_objects:
                scene_objects[scene_id] = []

            if object_id not in scene_objects[scene_id]:
                scene_objects[scene_id].append(object_id)

        return scene_objects
