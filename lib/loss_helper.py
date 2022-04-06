# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os

from icecream import ic

# sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from utils.nn_distance import nn_distance
from lib.config import CONF


def compute_cap_loss(data_dict, mode="gt"):
    """ Compute cluster caption loss

    Args:
        data_dict: dict (read-only)

    Returns:
        cap_loss, cap_acc
    """

    if mode == "gt":
        # unpack
        pred_caps = data_dict["lang_cap"]  # (B, num_words - 1, num_vocabs)
        des_lens = data_dict["lang_len"]  # batch_size
        num_words = des_lens.max()
        target_caps = data_dict["lang_ids"][:, 1:num_words]  # (B, num_words - 1)
        _, _, num_vocabs = pred_caps.shape

        # caption loss
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        cap_loss = criterion(pred_caps.reshape(-1, num_vocabs), target_caps.reshape(-1))

        if 'lang_cap_s' in data_dict:
            pred_caps_s = data_dict["lang_cap_s"]  # (B, num_words - 1, num_vocabs)
            cap_loss += criterion(pred_caps_s.reshape(-1, num_vocabs), target_caps.reshape(-1))
            cap_loss /= 2

        # caption acc
        pred_caps = pred_caps.reshape(-1, num_vocabs).argmax(-1)  # B * (num_words - 1)
        target_caps = target_caps.reshape(-1)  # B * (num_words - 1)
        masks = target_caps != 0
        masked_pred_caps = pred_caps[masks]
        masked_target_caps = target_caps[masks]
        cap_acc = (masked_pred_caps == masked_target_caps).sum().float() / masks.sum().float()
    elif mode == "votenet":
        # unpack
        pred_caps = data_dict["lang_cap"]  # (B, num_words - 1, num_vocabs)
        des_lens = data_dict["lang_len"]  # batch_size
        num_words = des_lens.max()
        target_caps = data_dict["lang_ids"][:, 1:num_words]  # (B, num_words - 1)

        _, _, num_vocabs = pred_caps.shape

        # caption loss
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        cap_loss = criterion(pred_caps.reshape(-1, num_vocabs), target_caps.reshape(-1))

        # mask out bad boxes
        good_bbox_masks = data_dict["good_bbox_masks"].unsqueeze(1).repeat(1, num_words - 1)  # (B, num_words - 1)
        good_bbox_masks = good_bbox_masks.reshape(-1)  # (B * num_words - 1)
        cap_loss = torch.sum(cap_loss * good_bbox_masks) / (torch.sum(good_bbox_masks) + 1e-6)

        num_good_bbox = data_dict["good_bbox_masks"].sum()
        if num_good_bbox > 0:  # only apply loss on the good boxes
            pred_caps = pred_caps[data_dict["good_bbox_masks"]]  # num_good_bbox
            target_caps = target_caps[data_dict["good_bbox_masks"]]  # num_good_bbox

            # caption acc
            pred_caps = pred_caps.reshape(-1, num_vocabs).argmax(-1)  # num_good_bbox * (num_words - 1)
            target_caps = target_caps.reshape(-1)  # num_good_bbox * (num_words - 1)
            masks = target_caps != 0
            masked_pred_caps = pred_caps[masks]
            masked_target_caps = target_caps[masks]
            cap_acc = (masked_pred_caps == masked_target_caps).sum().float() / masks.sum().float()
        else:  # zero placeholder if there is no good box
            cap_acc = torch.zeros(1)[0].cuda()

    return cap_loss, cap_acc


def radian_to_label(radians, num_bins=6):
    """
        convert radians to labels

        Arguments:
            radians: a tensor representing the rotation radians, (batch_size)
            radians: a binary tensor representing the valid masks, (batch_size)
            num_bins: number of bins for discretizing the rotation degrees

        Return:
            labels: a long tensor representing the discretized rotation degree classes, (batch_size)
    """

    boundaries = torch.arange(np.pi / num_bins, np.pi - 1e-8, np.pi / num_bins).cuda()
    labels = torch.bucketize(radians, boundaries)

    return labels


def get_loss(data_dict, mode="gt", use_rl=False):
    """ Loss functions
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """

    if not use_rl:
        cap_loss, cap_acc = compute_cap_loss(data_dict, mode)

        # store
        data_dict["cap_loss"] = cap_loss
        data_dict["cap_acc"] = cap_acc
    else:
        # store
        data_dict["cap_acc"] = torch.zeros(1)[0].cuda()

    # Final loss function
    loss = data_dict["cap_loss"]

    # loss *= 10 # amplify

    if 'kd_loss' in data_dict:
        kd_loss = data_dict['kd_loss']
    else:
        kd_loss = 0
        data_dict['kd_loss'] = torch.zeros(1)[0].cuda()

    data_dict["loss"] = loss + kd_loss

    return data_dict
