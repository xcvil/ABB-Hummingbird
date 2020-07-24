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

class ImageRename(object):
	"""docstring for ImageRename"""
	def __init__(self):
		super(ImageRename, self).__init__()
		self.path = '/media/xiaochenzheng/SSD/新建文件夹'


	def rename(self):

		filelist = os.listdir(self.path)
		total_num = len(filelist)

		i = 210
		for item in filelist:
		    if item.endswith('.jpg'):
		        src = os.path.join(os.path.abspath(self.path), item)
		        dst = os.path.join(os.path.abspath(self.path), '0000' + format(str(i), '0>3s') + '.jpg')
		        os.rename(src, dst)
		        print('converting %s to %s ...' % (src, dst))
		        i = i + 1
		print('total %d to rename & converted %d jpgs' % (total_num, i))

if __name__ == '__main__':
    newname = ImageRename()
    newname.rename()

