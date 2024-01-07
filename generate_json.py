import json
import random

# 从JSON文件加载类别
json_file_path = "datasets/nuscenes.json"
with open(json_file_path, "r") as json_file:
    categories = json.load(json_file)


color_interval = 256 // len(categories)

# 为每个类别生成颜色
categories_colors = {}
for i, category in enumerate(categories):
    # 计算RGB颜色
    red = i * color_interval
    green = (i * color_interval + 128) % 256  # 为了使颜色更加均匀，可以调整这里的算法
    blue = (i * color_interval + 64) % 256  # 同样可以调整这里的算法

    color = [red, green, blue]
    categories_colors[category] = color

# 保存结果到JSON文件
output_json_path = "datasets/generated_colors.json"
with open(output_json_path, "w") as json_file:
    json.dump(categories_colors, json_file, indent=4)


print(f"saved json in:{output_json_path}")
