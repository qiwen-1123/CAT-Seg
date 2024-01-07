# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

try:
    import detectron2
except ModuleNotFoundError:
    os.system("pip install git+https://github.com/facebookresearch/detectron2.git")

try:
    import segment_anything
except ModuleNotFoundError:
    os.system(
        "pip install git+https://github.com/facebookresearch/segment-anything.git"
    )

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    track,
)

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from cat_seg import add_cat_seg_config
from demo.predictor import VisualizationDemo
import gradio as gr
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as fc
import json
from nuscenes_seg.nuscenes_infer import nuscenes_infer

# constants
WINDOW_NAME = "CAT-Seg demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/demo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        default="datasets/nuscenes",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default="output/",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--seg_class",
        default="datasets/nuscenes.json",
        help="The json file contains segmented classes",
    )
    parser.add_argument(
        "--seg_mask",
        default="./datasets/generated_colors.json",
        help="The json file contains colors of masks",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=([]),
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    global demo

    demo = VisualizationDemo(cfg)

    input_path = args.input
    output_path = args.output
    seg_class = args.seg_class
    seg_mask = args.seg_mask

    nuscenes = nuscenes_infer(input_path, output_path, demo, seg_class, seg_mask)
    nuscenes.inference()
