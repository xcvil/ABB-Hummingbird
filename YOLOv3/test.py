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
import time

INPUT_SIZE   = 416
NUM_CLASS    = len(utils.read_class_names(cfg.YOLO.CLASSES))
CLASSES      = utils.read_class_names(cfg.YOLO.CLASSES)

predicted_dir_path = './mAP/predicted'
ground_truth_dir_path = './mAP/ground-truth'
if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
if os.path.exists(cfg.TEST.DECTECTED_IMAGE_PATH): shutil.rmtree(cfg.TEST.DECTECTED_IMAGE_PATH)

os.mkdir(predicted_dir_path)
os.mkdir(ground_truth_dir_path)
os.mkdir(cfg.TEST.DECTECTED_IMAGE_PATH)

# Build Model
input_layer  = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])
feature_maps = YOLOv3(input_layer)

bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)

model = tf.keras.Model(input_layer, bbox_tensors)
model.load_weights("./Arxic/GAUGE_new/gauge")

with open(cfg.TEST.ANNOT_PATH, 'r') as annotation_file:
    for num, line in enumerate(annotation_file):
        annotation = line.strip().split()
        image_path = annotation[0]
        image_name = image_path.split('/')[-1]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print('ccccccccccccccccccccccc')
        print(image.dtype)
        print(image.shape)
        image1 = image.astype(np.float32)
        print(image1.dtype)
        bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

        if len(bbox_data_gt) == 0:
            bboxes_gt=[]
            classes_gt=[]
        else:
            bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
        ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

        print('=> ground truth of %s:' % image_name)
        num_bbox_gt = len(bboxes_gt)
        with open(ground_truth_path, 'w') as f:
            for i in range(num_bbox_gt):
                class_name = CLASSES[classes_gt[i]]
                xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                f.write(bbox_mess)
                print('\t' + str(bbox_mess).strip())
        print('=> predict result of %s:' % image_name)
        predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
        # Predict Process
        image_size = image.shape[:2]
        image_data = utils.image_preporcess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        pred_bbox = model.predict(image_data)
        t2 = time.time()
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, image_size, INPUT_SIZE, cfg.TEST.SCORE_THRESHOLD)
        # prediction results
        bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')
        t3 = time.time()
        # [array([6.83409058e+02, 4.24715881e+02, 2.38848486e+03, 1.86276355e+03,
        #        9.61407304e-01, 0.00000000e+00]), array([7.41513489e+02, 2.46399756e+03, 2.51385986e+03, 3.03007666e+03,
        #        8.40067804e-01, 0.00000000e+00]), array([7.32385132e+02, 9.42967346e+02, 9.33952271e+02, 1.14114160e+03,
        #        9.65925813e-01, 1.00000000e+00]), array([7.82523376e+02, 1.24065161e+03, 8.86735779e+02, 1.33916272e+03,
        #        9.78054941e-01, 2.00000000e+00])]
        print(t2-t1)

        if cfg.TEST.DECTECTED_IMAGE_PATH is not None:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = utils.draw_bbox(image, bboxes)
            cv2.imwrite(cfg.TEST.DECTECTED_IMAGE_PATH+image_name, image)

        with open(predict_result_path, 'w') as f:
            for bbox in bboxes:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])
                class_name = CLASSES[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = list(map(str, coor))
                bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                f.write(bbox_mess)
                print('\t' + str(bbox_mess).strip())

