import argparse
import json
import yaml


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='config file')

    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
    parser.add_argument("--dataset", type=str, help="Choose a dataset: ScanRefer or Nr3D", default="ScanRefer")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--seed", type=int, default=400, help="random seed")

    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=20)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=40)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=2000)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)

    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_locals", type=int, default=-1, help="Number of local objects [default: -1]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")

    parser.add_argument("--criterion", type=str, default="cider",
                        help="criterion for selecting the best model [choices: bleu-1, bleu-2, bleu-3, bleu-4, cider, rouge, meteor, sum]")

    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_augment", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_caption", action="store_true", help="Do NOT train the caption module.")

    parser.add_argument("--use_tf", action="store_true", help="enable teacher forcing in inference.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_pretrained", type=str,
                        help="Specify the folder name containing the pretrained detection module.")

    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")

    parser.add_argument("--use_train", action="store_true", help="Use train split in evaluation.")
    parser.add_argument("--use_last", action="store_true", help="Use the last model")
    parser.add_argument("--force", action="store_true", help="generate the results by force")
    parser.add_argument("--save_interm", action="store_true", help="Save the intermediate results")
    parser.add_argument("--min_iou", type=float, default=0.25, help="Min IoU threshold for evaluation")
    

    args = parser.parse_args()

    with open(args.config_file, 'r') as fin:
        configs_dict = yaml.load(fin, Loader=yaml.FullLoader)
    apply_configs(args, configs_dict)

    return args

def apply_configs(args, config_dict):
    for key in config_dict:
        for k, v in config_dict[key].items():
            setattr(args, k, v)