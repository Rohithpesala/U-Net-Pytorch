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
from utils import *
from model import UNet

if HAVE_CUDA:
	import torch.cuda as cuda

def train(args):
	# Dataset loading
	# instantiate model
	# instantiate optimizer
	# call for an epoch
	prepare_env(args)
	model = load_model(os.getcwd()+args.output_dir)
	criterion = nn.CrossEntropyLoss() # Only for RELU model
	optimizer = loadOptimizer(model, os.getcwd())

	if HAVE_CUDA:
		criterion = criterion.cuda()
	
	train_start_time = time.time()
	epoch_start_time = train_start_time
	for i in range(args.num_epochs):
		# run model
		model = train_epoch(args,dataset_iterator)
		
		#Logging info
		end_time = time.time()
		epoch_duration = end_time-epoch_start_time
		epoch_start_time = end_time
		print "Epoch(",i,"/",num_epochs,") Duration: ", epoch_duration

	total_training_time = time.time()-train_start_time
	print "Total training time: ", total_training_time
	
	pass

def validation():
	pass

def test():
	pass

def train_epoch(args, iterator):
	