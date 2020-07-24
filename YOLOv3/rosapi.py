#! /usr/bin/env python
# coding=utf-8
"""Copyright (c) 2019 Xiaochen Zheng @ABB Group

Holders of copies of this code and documentation are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope ABB Group.

That is, no partial/full copy nor modification of this code should be made publicly or privately available to
other parties.

Contact: xzheng@ethz.ch; xzheng.eth@gmail.com
"""

import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov3 import YOLOv3, decode

# cv2 BGR WHC
use_GPU = True

if use_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def recognizeimage(width, height, image):

    imagematrix = np.zeros(shape=[height, width, 3], dtype=int)
    # initialize a numpy.ndarray to receive the image instead of using the cv2.imread op for ROS
    imagematrix = imagematrix.astype(np.uint8)
    # if using ROS, replace all image with imagematrix

    image = cv2.imread(image)  # TODO
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    '''the output type of cv2.imread() is numpy.ndarray, the data type of the output is numpy.ndarray.dtype = uint8
    print(image.dtype)
    uint8
    '''

    INPUT_SIZE   = 416
    NUM_CLASS    = len(utils.read_class_names(cfg.YOLO.CLASSES))
    CLASSES      = utils.read_class_names(cfg.YOLO.CLASSES)

    # Build Model
    input_layer  = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])
    feature_maps = YOLOv3(input_layer)

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, i)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.load_weights("./yolov3/model")

    # Predict Process
    image_size = image.shape[:2]
    image_data = utils.image_preporcess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    pred_bbox = model.predict(image_data)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = utils.postprocess_boxes(pred_bbox, image_size, INPUT_SIZE, cfg.TEST.SCORE_THRESHOLD)
    bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')

    image = utils.draw_bbox(image, bboxes)
    cv2.imwrite(cfg.TEST.DECTECTED_IMAGE_PATH, image)


