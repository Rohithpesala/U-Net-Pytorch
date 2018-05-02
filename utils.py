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

colors = loadmat('./mapping.mat')
colorsCity = colors['cityscapesMap']*255
nclasses = 35
labelCol = dict(zip(range(nclasses), colorsCity))

if HAVE_CUDA:
	import torch.cuda as cuda

def get_metrics(pred,truth):
	# calculate IOU and other metrics required
	_, pred = torch.max(pred,1)
	total_count = 1.0
	for i in truth.size():
		total_count *= i
	correct_count = torch.sum(pred.data == truth.data)
	accuracy = correct_count/total_count
	return accuracy

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
	save_path = os.path.join(os.getcwd(),args.output_dir,args.run_id,"save")
	if os.path.isdir(save_path):
		#load checkpoint
		filename = os.path.join(save_path,"last_checkpoint")
		print "model load"
		return torch.load(filename)
	else:
		if args.model_type == "unet":
			return UNet(args.kernel_size,args.pool_size,args.num_classes,args.pad_type)
		elif args.model_type == "mlp":
			return MLP(args.num_classes,args.pad_type)
		else:
			raise ValueError("model is not unet or mlp. Please specify correct model type")

def load_optimizer(args, model):
	optimizer = optim.Adam(model.parameters(), lr=0.01)
	save_path = os.path.join(os.getcwd(),args.output_dir,args.run_id,"save")
	if os.path.isdir(save_path):
		filename = os.path.join(save_path,"last_checkpoint_optim")
		print "optim_load"
		optimizer.load_state_dict(torch.load(filename))
	return optimizer

def save_model(args,model,best=False):
	checkpoint_name = "last_checkpoint"
	if best:
		checkpoint_name = "best_checkpoint"
	save_path = os.path.join(os.getcwd(),args.output_dir,args.run_id,"save",checkpoint_name)
	torch.save(model.cpu(),save_path)

def save_optimizer(args,optimizer,best=False):
	checkpoint_name = "last_checkpoint_optim"
	if best:
		checkpoint_name = "best_checkpoint_optim"
	save_path = os.path.join(os.getcwd(),args.output_dir,args.run_id,"save",checkpoint_name)
	torch.save(optimizer.state_dict(),save_path)


def save_data(args,model,optimizer,best = False):
	save_model(args,model,best)
	save_optimizer(args,optimizer,best)

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

def decode_labels(temp):
	red = temp.copy()
	green = temp.copy()
	blue = temp.copy()
	for l in range(0,nclasses):
	    red[temp == l] = labelCol[l][0]
	    green[temp == l] = labelCol[l][1]
	    blue[temp == l] = labelCol[l][2]
	rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
	rgb[:, :, 0] = red 
	rgb[:, :, 1] = green 
	rgb[:, :, 2] = blue
	return rgb

def reconstruct(lbl):
	lbl = decode_labels(lbl.numpy())
	imsave("./lable.png",lbl)

