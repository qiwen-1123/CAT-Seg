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
from rich.live import Live
import gradio as gr
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as fc
import json

import skimage.data

import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle


class nuscenes_infer:
    def __init__(self, input_path, output_path, demo, class_json_path, mask_json):
        self.input_path = input_path
        self.output_path = output_path
        self.mask_json = mask_json
        self.demo = demo

        with open(class_json_path, "r") as file:
            self.class_list = json.load(file)
            self.class_string = ",".join(self.class_list)

    def get_masks(self, preds, text):
        mask_json_filepath = self.mask_json

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
        num_img = 3

        cams = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ]

        for cam in cams:
            i = 0
            folder_path = os.path.join(self.input_path, "samples", cam)

            progress_inner = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
            )
            task = progress_inner.add_task(
                f"[cyan]Processing Images for {cam}",
                total=num_img,
                unit="image",
            )
            progress_inner.start()
            for filename in sorted(os.listdir(folder_path)):
                if filename.endswith((".jpg", ".jpeg", ".png")) and i < num_img:
                    i += 1
                    image_path = os.path.join(folder_path, filename)
                    image = cv2.imread(image_path)
                    image_name = filename[: filename.rfind(".")]

                    ### get predictions from CAT-SEG
                    outputs, predictions = self.predict(
                        image, self.class_string, model_type="CAT-Seg"
                    )
                    mask_image = self.get_masks(predictions, self.class_list)

                    ### selective search
                    selected_resions = self.selective_search_inference(
                        mask_image, predictions["sem_seg"]
                    )

                    ### save inference results
                    output_overlay_image_path = os.path.join(
                        self.output_path, "overlay", cam, f"{image_name}_overlay.jpg"
                    )
                    output_mask_image_path = os.path.join(
                        self.output_path, "mask", cam, f"{image_name}_mask.jpg"
                    )
                    output_raw_clip_path = os.path.join(
                        self.output_path, "pth", cam, f"{image_name}_catseg.pth"
                    )

                    output_ss_bbox_path = os.path.join(
                        self.output_path, "ss_bbox", cam, f"{image_name}_ss_box.pkl"
                    )

                    overlay_directory = os.path.dirname(output_overlay_image_path)
                    if not os.path.exists(overlay_directory):
                        os.makedirs(overlay_directory)

                    mask_directory = os.path.dirname(output_mask_image_path)
                    if not os.path.exists(mask_directory):
                        os.makedirs(mask_directory)

                    raw_clip_directory = os.path.dirname(output_raw_clip_path)
                    if not os.path.exists(raw_clip_directory):
                        os.makedirs(raw_clip_directory)

                    ss_bbox_directory = os.path.dirname(output_ss_bbox_path)
                    if not os.path.exists(ss_bbox_directory):
                        os.makedirs(ss_bbox_directory)

                    torch.save(predictions["sem_seg"], output_raw_clip_path)
                    cv2.imwrite(output_overlay_image_path, outputs)
                    cv2.imwrite(output_mask_image_path, mask_image)

                    with open(output_ss_bbox_path, "wb") as file:
                        pickle.dump(selected_resions, file)

                    progress_inner.update(task, advance=1)
                else:
                    break
            progress_inner.stop()

    def selective_search_inference(self, mask_img, raw_clip):
        img_rgb = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
        img_lbl, regions = selectivesearch.selective_search(
            img_rgb, scale=1000, sigma=0.8, min_size=100
        )
        selected_resions = regions[:50]

        print(selected_resions)
        # fig, ax = plt.subplots()
        # ax.imshow(img_rgb)

        # 添加选择性搜索的矩形框到图像上
        for r in selected_resions:
            x, y, w, h = r["rect"]
            cropped_region = raw_clip[:, y : y + h, x : x + w]
            sem_mean = torch.mean(cropped_region.float(), dim=(1, 2))
            cls_index = torch.argmax(sem_mean)
            r["labels"] = self.class_list[cls_index]

            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor="red", linewidth=1
            )
            # ax.add_patch(rect)

        # plt.show()

        return selected_resions
