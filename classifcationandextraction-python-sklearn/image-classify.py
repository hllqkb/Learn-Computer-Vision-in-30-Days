import os
import shutil
import random

def classify_images(source_dir, target_dir, train_ratio=0.8):
    """
    将图片分类到 train 和 val 文件夹，并按照类别存储。

    Args:
        source_dir (str): 原始图片所在的目录。
        target_dir (str): 目标根目录（包含 train 和 val 文件夹）。
        train_ratio (float): 训练集的比例，默认为 0.8（80%）。
    """
    # 定义类别关键词和对应的文件夹名称
    categories = {
        "cloudy": "cloudy",
        "rain": "rain",
        "shine": "shine",
        "sunrise": "sunrise"
    }

    # 创建目标文件夹结构
    for split in ["train", "val"]:
        for category in categories.values():
            os.makedirs(os.path.join(target_dir, split, category), exist_ok=True)

    # 遍历源文件夹中的图片
    for filename in os.listdir(source_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            # 检查图片属于哪个类别
            for keyword, category in categories.items():
                if keyword in filename.lower():
                    # 随机分配到 train 或 val
                    split = "train" if random.random() < train_ratio else "val"
                    # 源文件路径
                    src_path = os.path.join(source_dir, filename)
                    # 目标文件路径
                    dst_path = os.path.join(target_dir, split, category, filename)
                    # 复制文件
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied {filename} to {split}/{category}")
                    break

if __name__ == "__main__":
    # 设置源文件夹和目标文件夹
    source_directory = "./dataset2"  # 替换为你的图片文件夹路径
    target_directory = "./dataset"              # 目标文件夹名称

    # 调用函数进行分类
    classify_images(source_directory, target_directory)

    print("分类完成！")
