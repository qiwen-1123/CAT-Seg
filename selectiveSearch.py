import skimage.data
import cv2

import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle

# img = cv2.imread(
#     "output/mask/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558_mask.jpg"
# )
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_lbl, regions = selectivesearch.selective_search(
#     img_rgb, scale=1000, sigma=0.8, min_size=100
# )


# print(regions[:2])
# fig, ax = plt.subplots()
# ax.imshow(img_rgb)

# # 添加选择性搜索的矩形框到图像上
# for r in regions[:2]:
#     x, y, w, h = r["rect"]
#     rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=1)
#     ax.add_patch(rect)

# plt.show()

with open(
    "output/ss_bbox/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558_ss_box.pkl",
    "rb",
) as file:
    loaded_list = pickle.load(file)
print(type(loaded_list))
