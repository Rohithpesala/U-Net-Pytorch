<<<<<<< HEAD
from collections import OrderedDict
import os
import numpy as np
from sklearn.model_selection import train_test_split

def getgta5List(rootdir='.',suffix='',label=False):
	if label == True:
		return [os.path.join(looproot, filename)
			for looproot, _, filenames in os.walk(rootdir)
			for filename in filenames if filename.endswith(suffix)]
	else:
		return [os.path.join(looproot, filename)
        	for looproot, _, filenames in os.walk(rootdir)
        	for filename in filenames if filename.endswith(suffix)]

def splitData(labelIds,images):
	train_images,test_images,train_labels,test_labels = train_test_split(images,labelIds,test_size=0.2,random_state=42)
   	return train_labels,train_images,test_labels,test_images
=======
import PIL
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.autograd as ag
import torch.optim as optim
import numpy as np
import random
from torchvision import transforms
import sys

from constants import *
from model import UNet

if HAVE_CUDA:
	import torch.cuda as cuda

def get_metrics():
	# calculate IOU and other metrics required
	pass

def prepare_env(args):
	output_dir = args.output_dir
	run_id = args.run_id
	if os.path.isdir()
>>>>>>> 6d32e5df6009f551f3195d82ad3ffe1dbf6dade1
