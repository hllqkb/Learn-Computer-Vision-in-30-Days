import base64
import os
import urllib.request

import requests
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates as im_coordinates
import cv2
import numpy as np
import urllib.request

def wget(url, filename=None):
    if filename is None:
        filename = url.split('/')[-1]
    if not os.path.exists(filename):
        print(f"Downloading {url} to {filename} ...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")

# 下载模型和图片（如有需要）
model_path = './sam_vit_b_01ec64.pth'
if not os.path.exists(model_path):
    wget("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", model_path)

def remove_bg_with_sam(image: np.ndarray, x: int, y: int) -> np.ndarray:
    from segment_anything import SamPredictor, sam_model_registry
    import torch

    sam = sam_model_registry["vit_b"](checkpoint=model_path)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=np.array([[x, y]]),
        point_labels=np.array([1]),
        multimask_output=True
    )
    C, H, W = masks.shape
    result_mask = np.zeros((H, W), dtype=bool)
    for j in range(C):
        result_mask |= masks[j, :, :]
    result_mask = result_mask.astype(np.uint8)
    # 构造透明通道
    alpha_channel = np.ones(image.shape[:2], dtype=np.uint8) * 255
    alpha_channel[result_mask == 0] = 0
    # 直接生成 RGBA 格式
    bg_removed = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    bg_removed[:, :, 3] = alpha_channel
    return bg_removed

st.set_page_config(layout='wide')

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

local_path = os.path.dirname(os.path.abspath(__file__))
set_background(os.path.join(local_path, 'bg.jpg'))

api_endpoint = None

col01, col02 = st.columns(2)

# file uploader
file = col02.file_uploader('', type=['jpeg', 'jpg', 'png'])

# read image
if file is not None:
    image = Image.open(file).convert('RGB')
    image = image.resize((880, int(image.height * 880 / image.width)))

    # create buttons
    col1, col2 = col02.columns(2)

    # visualize image
    # click on image, get coordinates
    placeholder0 = col02.empty()
    with placeholder0:
        value = im_coordinates(image)
        if value is not None:
            print(value)

    if col1.button('Original', use_container_width=True):
        placeholder0.empty()
        placeholder1 = col02.empty()
        with placeholder1:
            col02.image(image, use_column_width=True)

    if col2.button('Remove background', type='primary', use_container_width=True):
        # call api
        placeholder0.empty()
        placeholder2 = col02.empty()

        if value is None:
            st.warning("请先在图片上点击一个前景点！")
        else:
            filename = '{}_{}_{}.png'.format(file.name, value['x'], value['y'])

            if os.path.exists(filename):
                result_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            else:
                np_image = np.array(image)
                if np_image.shape[2] == 4:
                    np_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2RGB)
                result_image = remove_bg_with_sam(np_image, value['x'], value['y'])
                cv2.imwrite(filename, result_image)

            with placeholder2:
                col02.image(result_image, use_column_width=True, channels="RGBA")
