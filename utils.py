from collections import OrderedDict
import os
import numpy as np
from sklearn.model_selection import train_test_split
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
import time

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
	output_path = os.path.join(os.getcwd(),output_dir)
	run_path = os.path.join(output_path,run_id)
	save_path = os.path.join(run_path,"save")
	log_path = os.path.join(run_path,"log")
	if not os.path.isdir(output_path):
		os.mkdir(output_path)
	if not os.path.isdir(run_path):
		os.mkdir(run_path)
		os.mkdir(save_path)
		os.mkdir(log_path)

def load_model(args):
	model_type = args.model_type
	save_path = os.path.join(os.getcwd(),args.run_id,"save")
	if os.path.isdir(save_path):
		#load checkpoint
		pass
	else:
		if args.model_type == "unet":
			return UNet(args.kernel_size,args.pool_size,args.num_classes,args.pad_type)
		elif args.model_type == "mlp":
			return MLP(args.num_classes,args.pad_type)
		else:
			raise ValueError("model is not unet or mlp. Please specify correct model type")

def load_optimizer(model, filename=None):
	optimizer = optim.Adam(model.parameters(), lr=0.01)
	if filename is not None and os.path.isfile(filename):
		optimizer.load_state_dict(torch.load(filename))


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

