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
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=([]),
        nargs=argparse.REMAINDER,
    )
    return parser


def get_masks(preds, text):
    mask_json_filepath = "./datasets/nuscenes_mask_dict.json"

    with open(mask_json_filepath, "r") as json_file:
        mask_dict = json.load(json_file)

    mask_image = np.zeros(
        (preds["sem_seg"].shape[1], preds["sem_seg"].shape[2], 3), dtype=np.uint8
    )
    preds = preds["sem_seg"].argmax(dim=0).cpu().numpy()  # C H W

    output_path = "output"
    dir_mask_all = os.path.join(output_path, "_mask_all.jpg")

    for i, t in enumerate(text):
        dir = os.path.join(output_path, f"{t}_mask.jpg")
        mask = preds == i
        mask_image[mask] = mask_dict[t]

    return mask_image


def predict(image, text, model_type):
    use_sam = model_type != "CAT-Seg"

    predictions, visualized_output = demo.run_on_image(image, text, use_sam)
    # save_masks(predictions, text.split(","))
    canvas = fc(visualized_output.fig)
    canvas.draw()
    out = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
        canvas.get_width_height()[::-1] + (3,)
    )

    return out[..., ::-1], predictions


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    global demo

    demo = VisualizationDemo(cfg)
    image_path = "nuscenes.jpg"
    image = cv2.imread(image_path)

    json_file_path = "datasets/nuscenes.json"

    with open(json_file_path, "r") as file:
        data_list = json.load(file)
        result_string = ",".join(data_list)

    input_path = "datasets/nuscenes"
    output_path = "output"

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            "[cyan]Processing Images", total=len(os.listdir(input_path)), unit="image"
        )

        for filename in os.listdir(input_path):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(input_path, filename)
                image = cv2.imread(image_path)
                image_name = filename[: filename.rfind(".")]

                outputs, predictions = predict(
                    image, result_string, model_type="CAT-Seg"
                )
                mask_image = get_masks(predictions, result_string.split(","))

                output_overlay_image_path = os.path.join(
                    output_path, f"{image_name}_overlay.jpg"
                )
                output_mask_image_path = os.path.join(
                    output_path, f"{image_name}_mask.jpg"
                )

                cv2.imwrite(output_overlay_image_path, outputs)
                cv2.imwrite(output_mask_image_path, mask_image)

                progress.update(task, advance=1)

    # for filename in track(os.listdir(input_path)):
    #     if filename.endswith((".jpg", ".jpeg", ".png")):
    #         image_path = os.path.join(input_path, filename)
    #         image = cv2.imread(image_path)
    #         outputs = predict(image, result_string, model_type="CAT-Seg")

    #         output_image_path = os.path.join(output_path, f"{filename}_output.jpg")
    #         cv2.imwrite(output_image_path, outputs)

    # ouputs = predict(
    #     image, result_string, model_type="CAT-Seg"
    # )
    # cv2.imshow("Image Visualization", ouputs)
    # key = cv2.waitKey(0)
    # if key != None:
    #     cv2.destroyAllWindows()
    #     exit()
