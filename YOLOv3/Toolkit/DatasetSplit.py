#! /usr/bin/env python
# coding=utf-8
"""Copyright (c) 2019 Xiaochen Zheng @ ETH Zurich. All rights reserved.

Holders of copies of this code and documentation are not allowed to copy, distribute or modify
any of the mentioned materials.

That is, no partial/full copy nor modification of this code should be made publicly or privately available to
other parties.

Contact: xzheng@ethz.ch; xzheng.eth@gmail.com
"""
import os
import random
import math

# anno_path = '/home/xiaochenzheng/data/AOD/AOD2019/Annotations/'
# img_path = '/home/xiaochenzheng/data/AOD/AOD2019/JPEGImages/'
# save_dir = '/home/xiaochenzheng/data/AOD/AOD2019/ImageSets/'

class DatasetSplit(object):
	"""docstring for DatasetSplit"""
	def __init__(self):
		super(DatasetSplit, self).__init__()
		self.anno_path = '/home/xiaochenzheng/data/GAUGE/GAUGE2019/Annotations/'
		self.img_path = '/home/xiaochenzheng/data/GAUGE/GAUGE2019/JPEGImages/'
		self.save_dir = '/home/xiaochenzheng/data/GAUGE/GAUGE2019/ImageSets/Main/'

		
	def get_namelist(self):

		imglist = os.listdir(self.img_path)
		annolist = os.listdir(self.anno_path)
		namelist = []
		for file in imglist:
			name, _ = os.path.splitext(file)
			namelist.append(name)

		return namelist, annolist, imglist

	def split_trainval(self, ratio_train, ratio_val, namelist):
		'''ratio_train: ratio of train set: [0, 1], 0.9 suggested without the test set
		namelist emmmmm name list 
		'''

		if ratio_train == 1:
			print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
			print('you need a validation set!!!')
			raise ValueError
		if (ratio_train+ratio_val) > 1:
			print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
			print('who taught you math!!!')
			raise ValueError
		if ratio_train < ratio_val:
			print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
			print('r u kidding me??? train set is smaller than val one???')
			raise ValueError

		ratio_test = 1 - ratio_train -ratio_val
		print('test set ratio:')
		print(ratio_test)
		num = len(namelist)
		num_train = int(round(ratio_train * num))
		num_val = int(round(ratio_val * num))
		print('the num of train and val set is:')
		print(num_train, num_val)
		random.shuffle(namelist)
		
		if ratio_test < math.e-4:
			trainset = namelist[:num_train]
			valset = namelist[num_train:]
			
			train = open(self.save_dir + 'train.txt', 'w')
			for file in trainset:
				train.write(file)
				train.write('\n')
			train.close()

			val = open(self.save_dir + 'val.txt', 'w')
			for file in valset:
				val.write(file)
				val.write('\n')
			val.close()

		elif ratio_test >= math.e-4:
			trainset = namelist[:num_train]
			valset = namelist[num_train: (num_train+num_val)]
			testset = namelist[(num_train+num_val):]

			train = open(self.save_dir + 'train.txt', 'w')
			for file in trainset:
				train.write(file)
				train.write('\n')
			train.close()

			val = open(self.save_dir + 'val.txt', 'w')
			for file in valset:
				val.write(file)
				val.write('\n')
			val.close()

			test = open(self.save_dir + 'test.txt', 'w')
			for file in testset:
				test.write(file)
				test.write('\n')
			test.close()

		else:
			raise ValueError

if __name__ == '__main__':

	DatasetSplit = DatasetSplit()
	namelist, _, _ = DatasetSplit.get_namelist()
	DatasetSplit.split_trainval(0.95, 0.05, namelist=namelist)
