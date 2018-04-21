import math
import numbers
import random
import numpy as np

from PIL import Image, ImageOps

class Scale(object):
	def __init__(self,size):
		self.size = size
	def __call__(self,img, mask):
		w,h = size
		img = Image.fromarray(img, mode ='RGB')
		mask = Image.fromarray(mask)
		img = img.resize((w,h),Image.BILINEAR)
		mask = mask.resize((w,h),Image.NEAREST)
		return np.array(img),np.array(mask,dtype=np.uint8)

# class Compose(object):
#     def __init__(self, augmentations):
#         self.augmentations = augmentations

#     def __call__(self, img, mask):
#         img, mask = Image.fromarray(img, mode='RGB'), Image.fromarray(mask, mode='L')            
#         import pdb
#         pdb.set_trace()
#         # print(img.size)
#         for a in self.augmentations:
#             img, mask = a(img, mask)
#         return np.array(img), np.array(mask, dtype=np.uint8)




# class Scale(object):
#     def __init__(self, size):
#         self.size = size

#     def __call__(self, img, mask):
#         img.size = mask.size
#         w, h = img.size
#         print(self.size)
#         print(w,h)
#         if (w >= h and w == self.size) or (h >= w and h == self.size):
#         	print("this")
#         	return img, mask
#         if w > h:
#         	print("this**")
#         	oh = int(self.size * h / w)
#         	return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
#         else:
#         	print("this****")
#         	oh = self.size
#         	ow = int(self.size * w / h)
#         	return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)

