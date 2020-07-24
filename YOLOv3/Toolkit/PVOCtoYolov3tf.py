#! /usr/bin/env python
# coding=utf-8
"""Copyright (c) 2019 Xiaochen Zheng @ETH Zurich

Holders of copies of this code and documentation are not allowed to copy, distribute or modify
any of the mentioned materials.

That is, no partial/full copy nor modification of this code should be made publicly or privately available to
other parties.

Contact: xzheng@ethz.ch; xzheng.eth@gmail.com
"""
import os
import xml.etree.ElementTree as ET

class PVOCtoYolov3tf(object):
    """docstring for PVOCtoYolov3tf"""
    def __init__(self):
        super (PVOCtoYolov3tf, self).__init__()
        self.root = '/home/xiaochenzheng/data/'
        self.image_sets = [('GAUGE', '2019', 'val'),('GAUGE', '2019', 'train')]
        
    def tf_yolo_dataset_generator(self):
        # generate the format of dataset which yolov3 requires, from the typical VOC dataset
        ids = list()
        for (dataset, year, process) in self.image_sets:
            rootpath = os.path.join(self.root, dataset, (dataset+year))
            train_set = open(dataset + '_' + process +'.txt', 'w')

            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', process + '.txt')):
                ids.append((rootpath, line.strip()))

            for i in ids:
                imgpath = os.path.join(i[0], 'JPEGImages', i[1] + '.jpg')
                train_set.write(imgpath)
                train_set.write(' ')
                annopath = os.path.join(i[0], 'Annotations', i[1] + '.xml')
                tree = ET.parse(annopath)
                anno_root = tree.getroot()

                for object in anno_root.iter('object'):
                    bndbox = object.find('bndbox')
                    xmin = bndbox.find('xmin').text.strip()
                    train_set.write(xmin)
                    train_set.write(',')
                    ymin = bndbox.find('ymin').text.strip()
                    train_set.write(ymin)
                    train_set.write(',')
                    xmax = bndbox.find('xmax').text.strip()
                    train_set.write(xmax)
                    train_set.write(',')
                    ymax = bndbox.find('ymax').text.strip()
                    train_set.write(ymax)
                    train_set.write(',')

                    # need to modify to your own labels
                    label = object.find('name').text.strip()
                    
                    if label == 'pointergauge':
                        train_set.write('0')
                    elif label == 'thermometer1':
                        train_set.write('1')
                    elif label == 'thermometer2':
                        train_set.write('2')
                    else:
                        raise ValueError('No such label!')
                    train_set.write(' ')

                train_set.write('\n')

            train_set.close()

if __name__ == '__main__':

    convert = PVOCtoYolov3tf()
    convert.tf_yolo_dataset_generator()