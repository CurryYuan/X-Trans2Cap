import os
import sys
import json
import torch
import numpy as np

from copy import deepcopy
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())  # HACK add the root folder

from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset import Dataset
from lib.config import CONF
from models.xtrans import TransformerCaptionModule
from lib.eval_helper import eval_cap
from in_out.arguments import parse_arguments

# constants
DC = ScannetDatasetConfig()


def get_dataloader(args, scanrefer, all_scene_list):
    dataset = Dataset(
        scanrefer=scanrefer,
        scanrefer_all_scene=all_scene_list,
        name=args.dataset,
        split='val',
        num_points=args.num_points,
        augment=False,
        use_color=args.use_color,
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return dataset, dataloader


def get_scannet_scene_list(data):
    # scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_{}.txt".format(split)))])
    scene_list = sorted(list(set([d["scene_id"] for d in data])))

    return scene_list


def get_eval_data(args):
    if args.dataset == "ScanRefer":
        scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
        scanrefer_val = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
    elif args.dataset == "Nr3d":
        scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_train.json")))
        scanrefer_val = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_val.json")))
    else:
        raise ValueError("Invalid dataset.")

    eval_scene_list = get_scannet_scene_list(scanrefer_train) if args.use_train else get_scannet_scene_list(
        scanrefer_val)
    scanrefer_eval = []
    for scene_id in eval_scene_list:
        data = deepcopy(scanrefer_train[0]) if args.use_train else deepcopy(scanrefer_val[0])
        data["scene_id"] = scene_id
        scanrefer_eval.append(data)

    print("eval on {} samples".format(len(scanrefer_eval)))

    return scanrefer_eval, eval_scene_list


def eval_caption(args):
    print("initializing...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get eval data
    scanrefer_eval, eval_scene_list = get_eval_data(args)

    # get dataloader
    dataset, dataloader = get_dataloader(args, scanrefer_eval, eval_scene_list)

    # get model
    model = TransformerCaptionModule(args, dataset)

    # load
    model_name = "model_last.pth" if args.use_last else "model.pth"
    model_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, model_name)
    model.load_state_dict(torch.load(model_path), strict=True)

    model.to(device)

    if args.use_train:
        pharse = 'train'
    else:
        pharse = 'val'

    # evaluate
    bleu, cider, rouge, meteor = eval_cap(args.mode, model, dataset, dataloader, pharse, args.use_pretrained, args.use_tf,
                                          force=args.force, save_interm=args.save_interm, min_iou=args.min_iou)

    # report
    print("\n----------------------Evaluation-----------------------")
    print("[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][0], max(bleu[1][0]), min(bleu[1][0])))
    print("[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][1], max(bleu[1][1]), min(bleu[1][1])))
    print("[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][2], max(bleu[1][2]), min(bleu[1][2])))
    print("[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][3], max(bleu[1][3]), min(bleu[1][3])))
    print("[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(cider[0], max(cider[1]), min(cider[1])))
    print("[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(rouge[0], max(rouge[1]), min(rouge[1])))
    print("[METEOR] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(meteor[0], max(meteor[1]), min(meteor[1])))
    print()


if __name__ == "__main__":
    args = parse_arguments()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpu)

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # evaluate
    eval_caption(args)
