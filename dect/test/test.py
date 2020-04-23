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

from dect.model.model import get_model

import torch
import torchvision

def getresult(img_path,outpath):
    NN_WEIGHT_FILE_PATH = 'dect/weight/efficient_rcnn_9.pth'

    VERSION_FAST = 49

    NMS_PARAM = 0.35

    CLASS_PROP_THR = 0.5

    RUN_MODE = "NMS"


    #img_path = "../data/Images_test/test.jpg"
    imge = Image.open(img_path).convert('RGB')
    testtransform = Compose([ToTensor()])
    img = testtransform(imge)
    model = get_model(VERSION_FAST)
    model.load_state_dict(torch.load(NN_WEIGHT_FILE_PATH))
    model.eval()

    print("Run Mode = ", RUN_MODE)

    if "NMS" == RUN_MODE:
        start = time.time()
        print(img.size())
        results = model([img])
        open_cv_image = np.array(imge)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        boxes = []
        for box, label, score, in zip(results[0]['boxes'], results[0]['labels'], results[0]["scores"]):
            boxes.append(box[:4].tolist() + [label] + [score])

        # boxes = np.array(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if boxes.shape[0] != 0:
            # keep = py_cpu_nms(boxes, 0.35)
            keep = box_ops.batched_nms(boxes[:, :4], boxes[:, 5], boxes[:, 4], NMS_PARAM)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            boxes = boxes[keep, :]

        #
        count=0
        for box in boxes:
            if box[5] < CLASS_PROP_THR:
                continue
            box = box.tolist()
            score = float(box[5])
            count+=1
            label_id = int(box[4]) - 1
            # label = CLASSES[label_id]
            label = 'Human'
            cv2.rectangle(open_cv_image,
                          (int(box[0]), int(box[1]), int(box[2]) - int(box[0]), int(box[3]) - int(box[1])),
                          (255, 225, 0), 2)
            cx = box[0]
            cy = box[1] + 12
            cv2.putText(open_cv_image, "{}:{:.2f}".format(label, score), (int(cx), int(cy)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
        # cv2.imshow("sd", open_cv_image)
        # cv2.imwrite("result/{}".format(img_path.split("/")[-1]), open_cv_image)
        # cv2.imshow("sd", open_cv_image)
        cv2.imwrite(outpath, open_cv_image)
        # cv2.waitKey(30000)
    else:
        start = time.time()
        print("img.size = ", img.size())
        results = model([img])
        open_cv_image = np.array(imge)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        print
        for box in results[0]['boxes']:
            box = box[:4].tolist()
            cv2.rectangle(open_cv_image,
                          (int(box[0]), int(box[1]), int(box[2]) - int(box[0]), int(box[3]) - int(box[1])),
                          (255, 225, 0), 2)
        # cv2.imshow("sd", open_cv_image)
        cv2.imwrite(outpath, open_cv_image)
        # cv2.waitKey(30000)
    return count

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))




