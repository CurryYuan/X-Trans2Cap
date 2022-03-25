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
    parser.add_argument("--num_graph_steps", type=int, default=0, help="Number of graph conv layer [default: 0]")

    parser.add_argument("--criterion", type=str, default="cider",
                        help="criterion for selecting the best model [choices: bleu-1, bleu-2, bleu-3, bleu-4, cider, rouge, meteor, sum]")

    parser.add_argument("--query_mode", type=str, default="center",
                        help="Mode for querying the local context, [choices: center, corner]")
    parser.add_argument("--graph_mode", type=str, default="edge_conv",
                        help="Mode for querying the local context, [choices: graph_conv, edge_conv]")
    parser.add_argument("--graph_aggr", type=str, default="add",
                        help="Mode for aggregating features, [choices: add, mean, max]")

    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_augment", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_caption", action="store_true", help="Do NOT train the caption module.")

    parser.add_argument("--use_tf", action="store_true", help="enable teacher forcing in inference.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_topdown", action="store_true", help="Use top-down attention for captioning.")
    parser.add_argument("--use_relation", action="store_true", help="Use object-to-object relation in graph.")
    parser.add_argument("--use_transformer", action="store_true", help="Use transformer module.")
    parser.add_argument("--use_orientation", action="store_true",
                        help="Use object-to-object orientation loss in graph.")
    parser.add_argument("--use_distance", action="store_true", help="Use object-to-object distance loss in graph.")
    parser.add_argument("--use_pretrained", type=str,
                        help="Specify the folder name containing the pretrained detection module.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")

    parser.add_argument("--debug", action="store_true", help="Debug mode.")

    args = parser.parse_args()

    with open(args.config_file, 'r') as fin:
        configs_dict = yaml.load(fin, Loader=yaml.FullLoader)
    apply_configs(args, configs_dict)

    return args

def apply_configs(args, config_dict):
    for key in config_dict:
        for k, v in config_dict[key].items():
            setattr(args, k, v)