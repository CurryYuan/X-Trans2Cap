import os
import sys
from easydict import EasyDict

CONF = EasyDict()

# path
CONF.PATH = EasyDict()
CONF.PATH.BASE = "/home/yuanzhihao/Projects/X-Trans2Cap/"  # TODO: change this
CONF.PATH.CLUSTER = "/mntntfs/med_data1/yuanzhihao/X-Trans2Cap/"  # TODO: change this
CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, 'data')
CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA, "scannet")
CONF.PATH.LIB = os.path.join(CONF.PATH.BASE, "lib")
CONF.PATH.MODELS = os.path.join(CONF.PATH.BASE, "models")
CONF.PATH.UTILS = os.path.join(CONF.PATH.BASE, "utils")

# append to syspath
for _, path in CONF.PATH.items():
    sys.path.append(path)

# scannet data
CONF.PATH.SCANNET_SCANS = os.path.join(CONF.PATH.SCANNET, "scans")
CONF.PATH.SCANNET_META = os.path.join(CONF.PATH.SCANNET, "meta_data")
CONF.PATH.SCANNET_DATA = os.path.join(CONF.PATH.SCANNET, "scannet_data")

# data
CONF.NYU40_LABELS = os.path.join(CONF.PATH.SCANNET_META, "nyu40_labels.csv")

# scannet
CONF.SCANNETV2_TRAIN = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_train.txt")
CONF.SCANNETV2_VAL = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_val.txt")
CONF.SCANNETV2_TEST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_test.txt")
CONF.SCANNETV2_LIST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2.txt")

# output
CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, "outputs")

# train
CONF.TRAIN = EasyDict()
CONF.TRAIN.MAX_DES_LEN = 30
CONF.TRAIN.SEED = 42
CONF.TRAIN.OVERLAID_THRESHOLD = 0.5
CONF.TRAIN.MIN_IOU_THRESHOLD = 0.25
CONF.TRAIN.NUM_BINS = 6

# eval
CONF.EVAL = EasyDict()
CONF.EVAL.MIN_IOU_THRESHOLD = 0.5

# data path
CONF.SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, "scannetv2-labels.combined.tsv")
CONF.VOCAB = os.path.join(CONF.PATH.DATA, "{}_vocabulary.json")  # dataset_name
CONF.GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")
CONF.VOCAB_WEIGHTS = os.path.join(CONF.PATH.DATA, "{}_vocabulary_weights.json")  # dataset_name
CONF.PATH.DATA_2D = os.path.join(CONF.PATH.DATA, "2d_feature_agg.npz")  # processed 2D features