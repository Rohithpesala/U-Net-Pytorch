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

from scipy.io import loadmat
from scipy.misc import *

from constants import *
from model import UNet

colors = loadmat('./mapping.mat')
colorsCity = colors['cityscapesMap']*255
nclasses = 35
labelCol = dict(zip(range(nclasses), colorsCity))
count = 0
if HAVE_CUDA:
	import torch.cuda as cuda

def get_metrics(args,pred,truth,infer=False):
	# calculate IOU and other metrics required
	_, pred = torch.max(pred,1)
	total_count = 1.0
	for i in truth.size():
		total_count *= i
	if infer:
		reconstruct(pred.data)
	correct_count = torch.sum(pred.data == truth.data)
	if pred.is_cuda:
		correct_count = correct_count.cpu().numpy()
	accuracy = correct_count/total_count
	IoU = compute_iou_batch(args,pred.data,truth.data)
	# print IoU
	return accuracy,IoU

def compute_iou_batch(args,pred,truth):
	iou = 0.0
	for i in range(pred.size()[0]):
		iou +=compute_iou_single(args,pred[i],truth[i])
	return iou/pred.size()[0]

def compute_iou_single(args,pred,truth):
	iou = 0.0
	# print pred
	# tot_c = 0
	for i in range(args.num_classes):
		pred_i = pred == i
		truth_i = truth == i
		# int_i = pred_i == truth_i
		# pred_i = pred_i.bool()
		# print type(pred_i)
		# print "pred",torch.sum(pred_i)
		# print "truth",torch.sum(truth_i)
		# print "int",torch.sum(int_i)
		tp = torch.sum((pred_i + truth_i) == 2)
		fpn = torch.sum((pred_i + truth_i) == 1)
		# print fpn,tp
		# tot_c+=torch.sum(pred_i)
		if pred.is_cuda:
			tp = tp.cpu().numpy()
			fpn = fpn.cpu().numpy()
		try:
			iou_temp = tp*1.0/(tp+fpn)
		except Exception as e:
			iou_temp = 0.0
		# print iou_temp
		iou += iou_temp
	# print "==========", tot_c
	return iou/args.num_classes

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
		try:
			print "model load"
			return torch.load(filename)
		except Exception as e:
			pass		
	if args.model_type == "unet":
		return UNet(args.kernel_size,args.pool_size,args.num_classes,args.pad_type,dropout = args.dropout)
	elif args.model_type == "mlp":
		return MLP(args.num_classes,args.pad_type)
	else:
		raise ValueError("model is not unet or mlp. Please specify correct model type")

def load_optimizer(args, model):
	optimizer = optim.Adam(model.parameters(), lr=0.01,weight_decay = args.weight_decay)
	save_path = os.path.join(os.getcwd(),args.output_dir,args.run_id,"save")
	if os.path.isdir(save_path):
		filename = os.path.join(save_path,"last_checkpoint_optim")
		try:
			print "optim_load"
			optimizer.load_state_dict(torch.load(filename))
		except Exception as e:
			pass		
	return optimizer

def save_model(args,model,best=False):
	checkpoint_name = "last_checkpoint"
	if best:
		checkpoint_name = "best_checkpoint"
	save_path = os.path.join(os.getcwd(),args.output_dir,args.run_id,"save",checkpoint_name)
	torch.save(model,save_path)

def save_optimizer(args,optimizer,best=False):
	checkpoint_name = "last_checkpoint_optim"
	if best:
		checkpoint_name = "best_checkpoint_optim"
	save_path = os.path.join(os.getcwd(),args.output_dir,args.run_id,"save",checkpoint_name)
	torch.save(optimizer.state_dict(),save_path)


def save_data(args,model,optimizer,best = False):
	save_model(args,model,best)
	save_optimizer(args,optimizer,best)

def getList(rootdir='.',suffix='',label=False):
	if label == True:
		return[os.path.join(looproot, filename)
			for looproot, _, filenames in os.walk(rootdir)
			for filename in filenames if filename.endswith(suffix)]
	else:
		return [os.path.join(looproot, filename)
        	for looproot, _, filenames in os.walk(rootdir)
        	for filename in filenames if filename.endswith(suffix)]
def getCityList(img_path,mask_path,suffix=''):
	images = []
	labelIds = []
	categories = os.listdir(img_path)
	for c in categories:
		c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_path, c))]
		for it in c_items:
			images.append(os.path.join(img_path, c, it + '_leftImg8bit.png'))
			labelIds.append(os.path.join(mask_path, c, it + suffix))
	return labelIds,images

def splitData(labelIds,images):
	train_images,test_images,train_labels,test_labels = train_test_split(images,labelIds,test_size=0.2,random_state=42)
   	return train_labels,train_images,test_labels,test_images

def decode_labels(temp):
	temp=np.transpose(temp,[1,0])
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

def reconstruct(lbl_batches):
	global count
	for lbl in lbl_batches:
		# print lbl,lbl[0:2]
		lbl1 = decode_labels(lbl.numpy())
		imsave("./lable"+str(count)+".png",lbl1)
		count += 1

def output_args(args):
	file_path = os.path.join(os.getcwd(),args.output_dir,args.run_id,"log","params.txt")
	log_file = open(file_path,"w")
	for arg in vars(args):
		print "%20s"%arg,"------", getattr(args,arg)
		print >> log_file,"%20s"%arg,"------", getattr(args,arg)

def load_args(args,params):
	for arg in vars(args):
		value = type(getattr(args,arg))(params[arg])
		setattr(args,arg,value) 
	# return args