import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import os
from utils import classify, set_background
# datasets
# https://data.mendeley.com/datasets/rscbjbr9sj/2
# 获取当前脚本所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 设置背景（使用绝对路径）
bg_path = os.path.join(current_dir, "bg", "bg1.jpg")
set_background(bg_path)

# 设置标题
st.title('Pneumonia Classification')
st.header('Please upload a chest X-ray image')

# 文件上传
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# 加载模型（带错误处理）
model_path = os.path.join(current_dir, "pneumonia_classifier.h5")
# 自定义反序列化函数，移除 'groups' 参数
def custom_depthwise_conv2d(**kwargs):
    if 'groups' in kwargs:
        del kwargs['groups']
    from keras.layers import DepthwiseConv2D
    return DepthwiseConv2D(**kwargs)

# 加载模型并传入自定义层
try:
    model = load_model(model_path, custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d})
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# 加载类别标签（带错误处理）
labels_path = os.path.join(current_dir, "labels.txt")
try:
    with open(labels_path, 'r') as f:
        class_names = [line.strip().split(' ')[1] for line in f.readlines()]
except Exception as e:
    st.error(f"Failed to load labels: {str(e)}")
    st.stop()

# 处理上传的图像
if file is not None:
    try:
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)

        # 分类图像
        class_name, conf_score = classify(image, model, class_names)

        # 显示结果
        st.write(f"## {class_name}")
        st.write(f"### Confidence: {conf_score*100:.1f}%")
        
        # 根据结果添加颜色提示
        if "pneumonia" in class_name.lower():
            st.error("Warning: Pneumonia detected! Please consult a doctor.")
        else:
            st.success("Normal chest X-ray detected")
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")