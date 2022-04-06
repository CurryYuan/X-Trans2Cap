import os
import sys
import json
import h5py
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from copy import deepcopy

sys.path.insert(0, os.getcwd())  # HACK add the root folder
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.solver import Solver
from lib.config import CONF
from models.xtrans import TransformerCaptionModule
from lib.dataset import Dataset
from in_out.arguments import parse_arguments

# constants
DC = ScannetDatasetConfig()


def get_dataloader(args, scanrefer, all_scene_list, split, augment):
    dataset = Dataset(
        scanrefer=scanrefer,
        scanrefer_all_scene=all_scene_list,
        name=args.dataset,
        split=split,
        num_points=args.num_points,
        augment=augment,
        use_color=args.use_color,
    )
    is_shuffle = True if split == 'train' else False
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=is_shuffle, num_workers=8, pin_memory=True)

    return dataset, dataloader


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params


def get_solver(args, dataset, dataloader):
    # initiate model
    model = TransformerCaptionModule(args, dataset["train"])  
    # to device
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.use_rl:
        model_path = os.path.join(CONF.PATH.OUTPUT, args.pretrained_path, "model.pth")
        model.load_state_dict(torch.load(model_path), strict=True)

    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag:
            stamp = args.tag
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    # scheduler parameters for training solely the detection pipeline
    LR_DECAY_STEP = [15, 20]
    LR_DECAY_RATE = 0.1
    BN_DECAY_STEP = None
    BN_DECAY_RATE = None

    solver = Solver(
        mode=args.mode,
        model=model,
        config=DC,
        dataset=dataset,
        dataloader=dataloader,
        optimizer=optimizer,
        stamp=stamp,
        val_step=args.val_step,
        use_tf=args.use_tf,
        use_rl=args.use_rl,
        lr_decay_step=LR_DECAY_STEP,
        lr_decay_rate=LR_DECAY_RATE,
        bn_decay_step=BN_DECAY_STEP,
        bn_decay_rate=BN_DECAY_RATE,
        criterion=args.criterion
    )
    num_params = get_num_params(model)
    print('params: ', num_params)

    return solver, num_params, root


def save_info(args, root, num_params, dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value

    info["num_train"] = len(dataset["train"])
    info["num_eval_train"] = len(dataset["eval"]["train"])
    info["num_eval_val"] = len(dataset["eval"]["val"])
    info["num_train_scenes"] = len(dataset["train"].scene_list)
    info["num_eval_train_scenes"] = len(dataset["eval"]["train"].scene_list)
    info["num_eval_val_scenes"] = len(dataset["eval"]["val"].scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)


def get_scanrefer(args):
    if args.dataset == "ScanRefer":
        scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
        scanrefer_eval_val = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
    elif args.dataset == "Nr3d":
        scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_train.json")))
        scanrefer_eval_val = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_val.json")))
    else:
        raise ValueError("Invalid dataset.")

    if args.debug:
        scanrefer_train = [scanrefer_train[0]]
        scanrefer_eval_val = [scanrefer_train[0]]

    train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
    val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_eval_val])))

    # eval
    scanrefer_eval_train = []
    for scene_id in train_scene_list:
        data = deepcopy(scanrefer_train[0])
        data["scene_id"] = scene_id
        scanrefer_eval_train.append(data)

    scanrefer_eval_val = []
    for scene_id in val_scene_list:
        data = deepcopy(scanrefer_train[0])
        data["scene_id"] = scene_id
        scanrefer_eval_val.append(data)

    print("train on {} samples from {} scenes".format(len(scanrefer_eval_train), len(train_scene_list)))
    print("eval on {} scenes from train and {} scenes from val".format(len(train_scene_list), len(val_scene_list)))

    return scanrefer_train, scanrefer_eval_train, scanrefer_eval_val, train_scene_list, val_scene_list


def train(args):
    # init training dataset
    print("preparing data...")
    scanrefer_train, scanrefer_eval_train, scanrefer_eval_val, train_scene_list, val_scene_list = get_scanrefer(args)

    # dataloader
    train_dataset, train_dataloader = get_dataloader(args, scanrefer_train, train_scene_list, "train", not args.no_augment)
    eval_train_dataset, eval_train_dataloader = get_dataloader(args, scanrefer_eval_train, train_scene_list, "train", False)
    eval_val_dataset, eval_val_dataloader = get_dataloader(args, scanrefer_eval_val, val_scene_list, "val", False)
    dataset = {
        "train": train_dataset,
        "eval": {
            "train": eval_train_dataset,
            "val": eval_val_dataset
        }
    }
    dataloader = {
        "train": train_dataloader,
        "eval": {
            "train": eval_train_dataloader,
            "val": eval_val_dataloader
        }
    }

    print("initializing...")
    solver, num_params, root = get_solver(args, dataset, dataloader)

    print("Start training...\n")
    save_info(args, root, num_params, dataset)
    solver(args.epoch, args.verbose)


if __name__ == "__main__":
    args = parse_arguments()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    train(args)
