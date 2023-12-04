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

import cv2
import numpy as np
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

import gradio as gr
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as fc
import json


class nuscenes_infer:
    def __init__(self, input_path, output_path, demo, class_json_path, mask_json):
        self.input_path = input_path
        self.output_path = output_path
        self.mask_json = mask_json
        self.demo = demo

        with open(class_json_path, "r") as file:
            data_list = json.load(file)
            self.result_string = ",".join(data_list)

    def get_masks(self, preds, text):
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

    def predict(self, image, text, model_type):
        use_sam = model_type != "CAT-Seg"

        predictions, visualized_output = self.demo.run_on_image(image, text, use_sam)
        canvas = fc(visualized_output.fig)
        canvas.draw()
        out = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
            canvas.get_width_height()[::-1] + (3,)
        )

        return out[..., ::-1], predictions

    def inference(
        self,
    ):
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                "[cyan]Processing Images",
                total=len(os.listdir(self.input_path)),
                unit="image",
            )

            for filename in os.listdir(self.input_path):
                if filename.endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(self.input_path, filename)
                    image = cv2.imread(image_path)
                    image_name = filename[: filename.rfind(".")]

                    outputs, predictions = self.predict(
                        image, self.result_string, model_type="CAT-Seg"
                    )
                    mask_image = self.get_masks(
                        predictions, self.result_string.split(",")
                    )

                    output_overlay_image_path = os.path.join(
                        self.output_path, f"{image_name}_overlay.jpg"
                    )
                    output_mask_image_path = os.path.join(
                        self.output_path, f"{image_name}_mask.jpg"
                    )

                    cv2.imwrite(output_overlay_image_path, outputs)
                    cv2.imwrite(output_mask_image_path, mask_image)

                    progress.update(task, advance=1)

                    # cv2.imshow("Image Visualization", ouputs)
                    # key = cv2.waitKey(0)
                    # if key != None:
                    #     cv2.destroyAllWindows()
                    #     exit()
