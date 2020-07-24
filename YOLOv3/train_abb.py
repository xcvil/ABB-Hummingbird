#! /usr/bin/env python
# coding=utf-8
"""Copyright (c) 2019 Xiaochen Zheng @ABB Group

Holders of copies of this code and documentation are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope ABB Group.

That is, no partial/full copy nor modification of this code should be made publicly or privately available to
other parties.

Contact: xzheng@ethz.ch; xzheng.eth@gmail.com
"""

import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
from core.dataset import Dataset
from core.yolov3 import YOLOv3, decode, compute_loss
from core.config import cfg

trainset = Dataset('train')
valset = Dataset('val')
logdir = "./data/log"
steps_per_epoch = len(trainset)
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch

input_tensor = tf.keras.layers.Input([416, 416, 3])
conv_tensors = YOLOv3(input_tensor)

output_tensors = []
for i, conv_tensor in enumerate(conv_tensors):
    pred_tensor = decode(conv_tensor, i)
    output_tensors.append(conv_tensor)
    output_tensors.append(pred_tensor)

model = tf.keras.Model(input_tensor, output_tensors)
optimizer = tf.keras.optimizers.Adam()
if os.path.exists(logdir): shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)
validate_writer = tf.summary.create_file_writer("./validate_log")
val_loss_list = []

for epoch in range(cfg.TRAIN.EPOCHS):
    for image_data, target in trainset:
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(3):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, *target[i], i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            tf.print("=> STEP %4d   EPOCH %4d   of %4d   in total   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, epoch + 1, cfg.TRAIN.EPOCHS,
                                                               optimizer.lr.numpy(),
                                                               giou_loss, conf_loss,
                                                               prob_loss, total_loss))
            # update learning rate
            global_steps.assign_add(1)
            if global_steps < warmup_steps:
                lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
            else:
                lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                )
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()

    val_tloss = float(0)
    val_itt = 0
    val_loss = float(0)
    for image_data, target in valset:
        val_itt += 1
        pred_result = model(image_data, training=False)
        giou_loss = conf_loss = prob_loss = 0

        # optimizing process
        for i in range(3):
            conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
            loss_items = compute_loss(pred, conv, *target[i], i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss
        val_tloss += total_loss
        tf.print("=> VALIDATION ON EPOCH %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
                 "prob_loss: %4.2f   total_loss: %4.2f" % (epoch + 1,
                                                           giou_loss, conf_loss,
                                                           prob_loss, total_loss))
        # writing summary data
        with validate_writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("validate_loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("validate_loss/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("validate_loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("validate_loss/prob_loss", prob_loss, step=global_steps)
        validate_writer.flush()

    val_loss = val_tloss / val_itt
    val_loss_list.append(val_loss)
    tf.print("==========> VALIDATION ON EPOCH %4d   average loss is %4.2f" %(epoch+1, val_loss))

    model.save_weights("./yolov3/gauge")
    # model.summary()
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()
    # open("converted_model.tflite", "wb").write(tflite_model)


val_loss_list1 = np.array(val_loss_list)
print("===================> The roadmap of the average validation loss is")
print(val_loss_list1)




