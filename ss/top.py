from model.models.common import DetectMultiBackend
from model.utils.general import check_img_size
from model.utils.datasets import LoadImages
import torch
import numpy as np
import cv2
from torch.nn import functional as F
from random import random

weights = 'model/trained-on-dummy-data.pt'

model = DetectMultiBackend(weights, device=torch.device('cpu'), dnn=False)
stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx

print(names)

# YOLOv3 is generally meant for object detection in an image,
# where there are RGB values in an X,Y plane.

# How do we translate I/Q data to this format in a way that makes sense?

# reading a 32x32 RGB image
# sample = cv2.imread('testimg.png')
# sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)  # cv2 default is BGR, switch to RGB
# (32, 32, 3)
# expected input size is (3, 32, 32), so we need to transpose the axes
# tensor = torch.from_numpy(sample.transpose(2, 0, 1)).to(model.device)
# change from bytes to floats (0.0-1.0)
# tensor = tensor.float() / 255

imgsz = check_img_size(640, s=stride)
dataset = LoadImages('model/frame1.jpg', img_size=imgsz, stride=stride, auto=pt and not jit)

for path, im, im0s, vid_cap, s in dataset:

    # tensor = torch.Tensor(1, 3, 32, 32)

    # usually done in batches where batch is the first dimension,
    # to conform to this tensor[None] will return tensor with size of (1, 3, 32, 32)
    im = torch.from_numpy(im).to(model.device)
    im = im.float() / 255
    if len(im.shape) == 3:
        result = model(im[None])
    else:
        result = model(im)

    count = {key: [0, 0] for key in model.names}

    for row in result[0]:
        center_x, center_y, width, height, object_score = row[:5]
        confidences = {name: row[5+i] for i, name in enumerate(model.names)}
        most_likely_classifier = max(confidences, key=lambda x: confidences[x])

        if object_score > 0.1:
            print(f"Detected object with {object_score:.6f} confidence @ ({int(center_x)},{int(center_y)}), size {int(width)}x{int(height)}")
            print(f"Confidences: {confidences}")
            print(f"Highest confidence: {most_likely_classifier}")

            # detection count
            count[most_likely_classifier][0] += 1
            # sum of confidences
            count[most_likely_classifier][1] += confidences[most_likely_classifier]

    mean_confidence = {key: (count[key][1]/count[key][0] if count[key][0] else 0) for key in count}
    print()
    print(f"Mean confidence values: {mean_confidence}")

    # print(result.shape)
    # Structure should be this, according these sources:
    # https://github.com/opencv/opencv/blob/4.x/samples/dnn/object_detection.cpp#L376-L447
    # https://towardsdatascience.com/yolo2-walkthrough-with-examples-e40452ca265f
    # rows are detected objects
    # column indices 0-3: center X, center Y, width, height
    # column index 4: object score (total confidence)
    # column index 5-7: confidence for each category (Radar, 5G, LTE)
    # print(result)

# Once the above is determined, we need a confidence threshold to decide whether
# a detection is adequately confident. However, that will probably require
# testing with ZeroMQ data to determine.

# y_hat = F.log_softmax(result, dim=1).argmax(dim=1)
# print(y_hat)