import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.capeval.cider.cider import Cider
from lib.config import CONF
from lib.eval_helper import decode_caption
from models.utils import PointNetPP, get_siamese_features
from transformer.m2_transformer import DualM2Transformer, M2Transformer

DC = ScannetDatasetConfig()
SCANREFER_ORGANIZED = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_organized.json")))


class TransformerCaptionModule(nn.Module):
    def __init__(self, args, dataset, emb_size=300, feat_size=128, hidden_size=512):
        
        super().__init__()
        self.vocabulary = dataset.vocabulary
        self.embeddings = dataset.glove
        self.num_vocabs = len(self.vocabulary["word2idx"])

        self.emb_size = emb_size
        self.feat_size = feat_size 
        self.hidden_size = hidden_size
        self.num_proposals = args.num_proposals
        self.num_class = DC.num_class
        self.use_rl = args.use_rl
        self.dataset = dataset

        # Pointnet++ Backbone
        self.object_encoder = PointNetPP(sa_n_points=[32, 16, None],
                                         sa_n_samples=[32, 32, None],
                                         sa_radii=[0.2, 0.4, None],
                                         sa_mlps=[[3, 64, 128],
                                                  [128, 128, 128, 256],                                 
                                                  [256, 256, 512, feat_size]],
                                         bn=False)

        self.captioner_teacher = DualM2Transformer(self.vocabulary, CONF.TRAIN.MAX_DES_LEN + 2, feat_size+64+self.num_class,
                                       padding_idx=0)
        self.captioner = M2Transformer(self.vocabulary, CONF.TRAIN.MAX_DES_LEN + 2, feat_size + 64 + self.num_class,
                                       padding_idx=0)
        self.fc_rpe = nn.Sequential(nn.Linear(6, 32),
                                    nn.LayerNorm(32))

        self.fc_ape = nn.Sequential(nn.Linear(6, 32),
                                    nn.LayerNorm(32))

        self.fc_2d = nn.Sequential(nn.Linear(2048, 512),
                                   nn.ReLU(),
                                   nn.LayerNorm(512),
                                   nn.Linear(512, feat_size))

        self.fc_bbox_2d = nn.Sequential(nn.Linear(4, 32),
                                        nn.LayerNorm(32))

        # transform the visual signals
        self.map_feat = nn.Sequential(
            nn.Linear(feat_size, feat_size),
            nn.ReLU()
        )

    def forward(self, data_dict, use_tf=True, is_eval=False, max_len=CONF.TRAIN.MAX_DES_LEN):
        if is_eval:
            data_dict = self.forward_scene_batch(data_dict, use_tf, max_len)
        elif self.use_rl:
            data_dict= self.forward_rl_batch(data_dict, use_tf, max_len)
        else:
            data_dict = self.forward_sample_batch(data_dict, max_len)

        return data_dict

    def forward_sample_batch(self, data_dict, max_len=CONF.TRAIN.MAX_DES_LEN, min_iou=CONF.TRAIN.MIN_IOU_THRESHOLD):
        """
        generate descriptions based on input tokens and object features
        """

        # unpack
        des_lens = data_dict["lang_len"]  # batch_size
        num_words = des_lens.max()
        target_caps = data_dict["lang_ids"][:, :num_words]  # (B, num_words)
        batch_size = des_lens.shape[0]
        bbox_mask = data_dict["bbox_mask"].unsqueeze(-1)

        # find the target object ids
        target_ids = data_dict["bbox_idx"]  # batch_size
        target_ious = torch.ones(batch_size).cuda()

        instance_points = data_dict['objects']  # (B, n_object, n_points, 6)
        # Get features for each segmented scan object based on color and point-cloud
        obj_feats = get_siamese_features(self.object_encoder, instance_points,
                                                aggregator=torch.stack)  # B X N_Objects x object-latent-dim
        sem_cls = data_dict['sem_cls_label']  # (batch_size, num_proposals)s)

        device = obj_feats.device
        sem_cls_one_hot = torch.zeros((batch_size, self.num_proposals, self.num_class), device=device)
        sem_cls_one_hot.scatter_(2, sem_cls.unsqueeze(-1), 1)  # src==1 so it"s *one-hot* (B,K,num_size_cluster)

        # NOTE when the IoU of best matching predicted boxes and the GT boxes
        # are smaller than the threshold, the corresponding predicted captions
        # should be filtered out in case the model learns wrong things
        good_bbox_masks = target_ious > min_iou  # batch_size
        num_good_bboxes = good_bbox_masks.sum()

        mean_target_ious = target_ious[good_bbox_masks].mean() if num_good_bboxes > 0 else torch.zeros(1)[0].cuda()

        pred_obbs = data_dict['bbox_obb']  # (batch_size, num_proposals, 7)
        ape = self.fc_ape(pred_obbs[:, :, :6])  # (B, n_object, d)

        # select object features
        # (batch_size, 7)
        target_obbs = torch.gather(pred_obbs, 1, target_ids.view(batch_size, 1, 1).repeat(1, 1, 7))
        rpe = torch.cat((pred_obbs[..., :3] - target_obbs[..., :3], pred_obbs[..., 3:6] / target_obbs[..., 3:6]),
                        dim=-1)
        rpe = self.fc_rpe(rpe)

        feat_2d = data_dict['feat_2d']
        bbox_2d = data_dict['bbox_2d']

        feat_2d = self.fc_2d(feat_2d)
        bbox_2d = self.fc_bbox_2d(bbox_2d)
        feat_2d = torch.cat((feat_2d, bbox_2d, sem_cls_one_hot, rpe), dim=-1)
        feat_2d = feat_2d * bbox_mask

        obj_feats = torch.cat((obj_feats, sem_cls_one_hot, ape, rpe), dim=-1)
        obj_feats = obj_feats * bbox_mask
        # obj_feats = torch.cat((obj_feats, feat_2d), dim=1)

        outputs_s, inter_feat_s, encoder_feat = self.captioner(obj_feats, target_caps)
        outputs, inter_feat_t = self.captioner_teacher(feat_2d, encoder_feat, target_caps)  # (batch_size, max_len, num_vocabs)

        criterion = nn.MSELoss()
        kd_loss = criterion(inter_feat_s, inter_feat_t)

        # store
        data_dict["lang_cap"] = outputs[:, :-1]
        data_dict['lang_cap_s'] = outputs_s[:, :-1]
        data_dict["pred_ious"] = mean_target_ious
        data_dict["good_bbox_masks"] = good_bbox_masks
        data_dict['kd_loss'] = kd_loss

        return data_dict

    def forward_rl_batch(self, data_dict, use_tf=False, max_len=CONF.TRAIN.MAX_DES_LEN):
        """
        generate descriptions based on input tokens and object features
        """

        # unpack
        des_lens = data_dict["lang_len"]  # batch_size
        num_words = des_lens.max()
        target_caps = data_dict["lang_ids"][:, :num_words]  # (B, num_words)
        batch_size = des_lens.shape[0]
        bbox_mask = data_dict["bbox_mask"].unsqueeze(-1)

        target_ids = data_dict["bbox_idx"]  # batch_size
        target_ious = torch.ones(batch_size).cuda()

        instance_points = data_dict['objects']  # (B, n_object, n_points, 6)
        # Get features for each segmented scan object based on color and point-cloud
        obj_feats = get_siamese_features(self.object_encoder, instance_points,
                                                aggregator=torch.stack)  # B X N_Objects x object-latent-dim
        sem_cls = data_dict['sem_cls_label']  # (batch_size, num_proposals)

        device = obj_feats.device
        sem_cls_one_hot = torch.zeros((batch_size, self.num_proposals, self.num_class), device=device)
        sem_cls_one_hot.scatter_(2, sem_cls.unsqueeze(-1), 1)  # src==1 so it"s *one-hot* (B,K,num_size_cluster)

        # NOTE when the IoU of best matching predicted boxes and the GT boxes
        # are smaller than the threshold, the corresponding predicted captions
        # should be filtered out in case the model learns wrong things
        good_bbox_masks = target_ious > CONF.TRAIN.MIN_IOU_THRESHOLD  # batch_size
        num_good_bboxes = good_bbox_masks.sum()

        mean_target_ious = target_ious[good_bbox_masks].mean() if num_good_bboxes > 0 else torch.zeros(1)[0].cuda()

        pred_obbs = data_dict['bbox_obb']  # (batch_size, num_proposals, 7)
        ape = self.fc_ape(pred_obbs[:, :, :6])  # (B, n_object, d)

        # select object features
        # (batch_size, 7)
        target_obbs = torch.gather(pred_obbs, 1, target_ids.view(batch_size, 1, 1).repeat(1, 1, 7))
        rpe = torch.cat((pred_obbs[..., :3] - target_obbs[..., :3], pred_obbs[..., 3:6] / target_obbs[..., 3:6]),
                        dim=-1)
        rpe = self.fc_rpe(rpe)

        feat_2d = data_dict['feat_2d']
        bbox_2d = data_dict['bbox_2d']

        feat_2d = self.fc_2d(feat_2d)
        bbox_2d = self.fc_bbox_2d(bbox_2d)
        feat_2d = torch.cat((feat_2d, bbox_2d, sem_cls_one_hot, rpe), dim=-1)
        feat_2d = feat_2d * bbox_mask

        obj_feats = torch.cat((obj_feats, sem_cls_one_hot, ape, rpe), dim=-1)
        obj_feats = obj_feats * bbox_mask
        # obj_feats = torch.cat((obj_feats, feat_2d), dim=1)

        beam_size = 5
        gts = {}
        gen = {}

        output, log_probs = self.captioner.beam_search(obj_feats, max_len + 2, beam_size=beam_size,
                                                       out_size=beam_size)[:2]

        dataset_ids = data_dict["dataset_idx"]
        for batch_id in range(batch_size):
            dataset_idx = dataset_ids[batch_id].item()
            scene_id = self.dataset.scanrefer[dataset_idx]["scene_id"]
            object_id = str(data_dict['object_id'][batch_id].item())
            corpus = [' '.join(x['token']) for x in SCANREFER_ORGANIZED[scene_id][object_id].values()]

            for i in range(5):
                pred = decode_caption(output[batch_id, i], self.vocabulary["idx2word"])
                gen.update({f'{dataset_idx}_{batch_id}_{i}': [pred]})
                gts.update({f'{dataset_idx}_{batch_id}_{i}': corpus})

        cider = Cider()
        reward = cider.compute_score(gts, gen)[1]
        reward = torch.from_numpy(reward).to(obj_feats.device).view(obj_feats.shape[0], beam_size)
        reward_baseline = torch.mean(reward, -1, keepdim=True)
        loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)
        loss = loss.mean()

        # store
        data_dict["pred_ious"] = mean_target_ious
        data_dict["good_bbox_masks"] = good_bbox_masks
        data_dict['cap_loss'] = loss

        return data_dict


    def forward_scene_batch(self, data_dict, use_tf=False, max_len=CONF.TRAIN.MAX_DES_LEN):
        """
        generate descriptions based on input tokens and object features
        """

        # unpack
        des_lens = data_dict["lang_len"]  # batch_size
        batch_size = des_lens.shape[0]

        instance_points = data_dict['objects']  # (B, n_object, n_points, 6)
        # Get features for each segmented scan object based on color and point-cloud
        obj_feats = get_siamese_features(self.object_encoder, instance_points,
                                                aggregator=torch.stack)  # B X N_Objects x object-latent-dim
        sem_cls = data_dict['sem_cls_label']  # (batch_size, num_proposals)

        device = obj_feats.device
        sem_cls_one_hot = torch.zeros((batch_size, self.num_proposals, self.num_class), device=device)
        sem_cls_one_hot.scatter_(2, sem_cls.unsqueeze(-1), 1)  # src==1 so it"s *one-hot* (B,K,num_size_cluster)
        bbox_mask = data_dict["bbox_mask"].unsqueeze(-1)

        # transform the features
        pred_obbs = data_dict['bbox_obb']  # (batch_size, num_proposals, 7)
        ape = self.fc_ape(pred_obbs[:, :, :6])  # (B, n_object, d)

        feat_2d = data_dict['feat_2d']
        bbox_2d = data_dict['bbox_2d']

        feat_2d = self.fc_2d(feat_2d)
        bbox_2d = self.fc_bbox_2d(bbox_2d)

        # recurrent from 0 to max_len - 2
        object_feats_flat = []
        feats_2d_flat = []
        for prop_id in range(self.num_proposals):
            # select object features
            target_obbs = pred_obbs[:, prop_id].unsqueeze(1)  # (batch_size, 1, 7)
            rpe = torch.cat((pred_obbs[..., :3] - target_obbs[..., :3], pred_obbs[..., 3:6] / target_obbs[..., 3:6]),
                            dim=-1)
            rpe = self.fc_rpe(rpe)

            object_feats = torch.cat((obj_feats, sem_cls_one_hot, ape, rpe), dim=-1)
            # temp_feat_2d = torch.cat((obj_feats, feat_2d, bbox_2d, sem_cls_one_hot, ape, rpe), dim=-1)
            object_feats = object_feats * bbox_mask
            # temp_feat_2d = temp_feat_2d * bbox_mask

            # object_feats = torch.cat((object_feats, temp_feat_2d), dim=1)
            object_feats_flat.append(object_feats)
            # feats_2d_flat.append(temp_feat_2d)

        object_feats_flat = torch.cat(object_feats_flat, dim=0)  # (num_proposals*batch_size, num_proposals, feat_size)
        object_feats_flat = object_feats_flat.reshape(self.num_proposals, batch_size, self.num_proposals,
                                                      -1).transpose(1, 0)
        object_feats_flat = object_feats_flat.reshape(batch_size*2, self.num_proposals//2, self.num_proposals, -1)

        # feats_2d_flat = torch.cat(feats_2d_flat, dim=0)  # (num_proposals*batch_size, num_proposals, feat_size)
        # feats_2d_flat = feats_2d_flat.reshape(self.num_proposals, batch_size, self.num_proposals, -1).transpose(1, 0)
        # feats_2d_flat = feats_2d_flat.reshape(batch_size*2, self.num_proposals//2, self.num_proposals, -1)

        outputs = []
        for i in range(object_feats_flat.shape[0]):
            output = self.captioner.beam_search(object_feats_flat[i], max_len + 2, beam_size=1, out_size=1)[0]
            # (num_proposals, max_len)
            # output = self.captioner_teacher.beam_search((feats_2d_flat[i], output), max_len + 2, beam_size=1, out_size=1)[0]
            dummy_probs = torch.zeros((output.shape[0], output.shape[1], self.num_vocabs), device=device)
            dummy_probs = dummy_probs.scatter(2, output.unsqueeze(2), 1)
            outputs.append(dummy_probs[:, :-1].unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)  # batch_size, num_proposals, num_words - 1, num_vocabs
        outputs = outputs.reshape(batch_size, self.num_proposals, -1, self.num_vocabs)

        # store
        data_dict["lang_cap"] = outputs

        return data_dict
