

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch

from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter, Resize
from torch.utils.data import DataLoader

from torchvision.ops import boxes as box_ops
from PIL import Image
import torch.nn.functional as F
import time
import numpy as np
import cv2

from model.model import get_model

import torch
import torchvision


NN_WEIGHT_FILE_PATH = 'D:\model\efficient_rcnn_9.pth'
torch_model = get_model(49)
torch_model.load_state_dict(torch.load(NN_WEIGHT_FILE_PATH))
#torch_model = torch.load(NN_WEIGHT_FILE_PATH) # pytorch模型加载
batch_size = 1  #批处理大小
input_shape = (3,244,244)   #输入数据

# set the model to inference mode
torch_model.eval()

x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = torch_model(x)
 # optionally, if you want to export the model to ONNX:
torch.onnx.export(torch_model, x, "faster_rcnn.onnx", opset_version = 11)


'''
x = torch.randn(batch_size,*input_shape)		# 生成张量
export_onnx_file = "test.onnx"					# 目的ONNX文件名
torch.onnx.export(InPlaceIndexedAssignmentONNX(),
                    x,
                    export_onnx_file,
                    opset_version=10,
                    do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=["input"],		# 输入名
                    output_names=["output"],	# 输出名
                   )


'''
'''
# Trace-based only

class LoopModel(torch.nn.Module):
    def forward(self, x, y):
        for i in range(y):
            x = x + i
        return x

model = LoopModel()
dummy_input = torch.ones(2, 3, dtype=torch.long)
loop_count = torch.tensor(5, dtype=torch.long)

torch.onnx.export(model, (dummy_input, loop_count), 'loop.onnx', verbose=True)
'''