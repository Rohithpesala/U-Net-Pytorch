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
	logger = ""
	model = load_model(args)
	criterion = nn.CrossEntropyLoss() # Only for RELU model
	optimizer = load_optimizer(model, os.getcwd())

	if HAVE_CUDA:
		criterion = criterion.cuda()
	
	total_training_loss = 0.0

	train_start_time = time.time()
	epoch_start_time = train_start_time
	for i in range(args.num_epochs):
		# run model
		model, loss = train_epoch(args,dataset_iterator)
		
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
	pass