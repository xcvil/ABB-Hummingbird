import sys
ros_path='/opt/ros/melodic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import cv2

sys.path.append(ros_path)

import rospy
from mavic_code.msg import ImageRGB
import numpy as np
import os
import shutil
import tensorflow as tf
from matplotlib import pyplot as PLT
import time
import core.utils as utils
from core.config import cfg
from core.yolov3 import YOLOv3,decode
from threading import Thread


class RecognizeProgram:
    model = 0
    INPUT_SIZE = 416
    imagematrix = 0
    number = 0

    def __init__(self):
        starttime = time.time()
        NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
        CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)

        #build model
        input_layer = tf.keras.layers.Input([self.INPUT_SIZE,self.INPUT_SIZE,3])
        feature_maps = YOLOv3(input_layer)

        bbox_tensors = []
        for i,fm in enumerate(feature_maps):
            bbox_tensor = decode(fm,i)
            bbox_tensors.append(bbox_tensor)

        self.model = tf.keras.Model(input_layer,bbox_tensors)
        self.model.load_weights("./yolov3/model")
        endtime = time.time()
        print("initialize the net: "+str(endtime-starttime))


    def callback(self,data):
        rospy.loginfo("Image Received. Size: [" + str(data.width)+"*"+str(data.height)+"]")
        starttime = time.time()
        image = np.reshape(data.image,(data.height,data.width))
        imagematrix = [image&0xFF,(image>>8)&0xFF,(image>>16)&0xFF]
        imagematrix = np.transpose(imagematrix,(1,2,0))
        self.number += 1
        print("Picture: " + str(self.number))
        save_path = '/home/viki/halloviki/'+str(self.number)+'-origin-image.jpg'
        cv2.imwrite(save_path,imagematrix)
        imagematrix = cv2.imread(save_path)
        endtime = time.time()
        print("Decode the Image: " + str(endtime-starttime))

        #predict the image
        starttime = time.time()
        imagematrix = imagematrix.astype(np.uint8)
        #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        image_size = imagematrix.shape[:2]
        image_data = utils.image_preporcess(np.copy(imagematrix),[self.INPUT_SIZE,self.INPUT_SIZE])
        image_data = image_data[np.newaxis,...].astype(np.float32)

        pred_bbox = self.model.predict(image_data)
        pred_bbox = [tf.reshape(x,(-1,tf.shape(x)[-1]))for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox,axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox,image_size,self.INPUT_SIZE,cfg.TEST.SCORE_THRESHOLD)
        bboxes = utils.nms(bboxes,cfg.TEST.IOU_THRESHOLD,method='nms')
        endtime = time.time()
        print("run the net: " + str(endtime-starttime))

        starttime = time.time()
        imagematrix = utils.draw_bbox(imagematrix,bboxes)
        cv2.imwrite('/home/viki/halloviki/'+str(self.number)+'-recognized-image.jpg',imagematrix)
        endtime = time.time()
        print("saved the image: "+str(endtime-starttime))
        

def recognizeImage2():
    recognizePro = RecognizeProgram()
    print("The Net is ready. ")
    rospy.init_node("recognize2",anonymous=True)
    rospy.Subscriber("imagefromAndroid",ImageRGB,recognizePro.callback)
    rospy.spin()

if __name__ == '__main__':
    recognizeImage2()