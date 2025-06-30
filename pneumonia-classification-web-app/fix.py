import keras
from keras.models import load_model, save_model
from keras.layers import DepthwiseConv2D
import os
local_path = os.path.dirname(os.path.abspath(__file__))

# 加载模型
model_path = os.path.join(local_path, "pneumonia_classifier.h5")
model = load_model(model_path)

# 递归地移除 `groups` 参数
def remove_groups_parameter(model):
    for layer in model.layers:
        if isinstance(layer, DepthwiseConv2D):
            if 'groups' in layer.__dict__:
                del layer.__dict__['groups']
            if 'config' in layer.__dict__ and 'groups' in layer.get_config():
                layer_config = layer.get_config()
                del layer_config['groups']
                layer.__init__(**layer_config)
        if hasattr(layer, 'layers') and layer.layers:
            remove_groups_parameter(layer)

remove_groups_parameter(model)

# 重新保存模型
new_model_path = os.path.join(local_path, "pneumonia_classifier_fixed.h5")
save_model(model, new_model_path)
