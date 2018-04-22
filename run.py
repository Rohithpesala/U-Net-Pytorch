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
from dataReader import *


if HAVE_CUDA:
	import torch.cuda as cuda

def train(args):
	
	# Dataset loading and preparing env
	prepare_env(args)
	all_datasets = dataReader(args)
	train_data = all_datasets['train']
	validation_data = all_datasets['validation']
	logger = ""
	best_train_loss = float("Inf")
	best_validation_loss = float("Inf")
	best_epoch = -1

	# instantiate model and optimizer
	model = load_model(args)
	model.train()
	criterion = nn.CrossEntropyLoss()
	optimizer = load_optimizer(args,model)
	if HAVE_CUDA:
		criterion = criterion.cuda()
	
	# log info
	train_start_time = time.time()
	epoch_start_time = train_start_time
	for i in range(args.num_epochs):
		
		# run model
		model, optimizer, present_train_loss = train_epoch(args,train_data,model,optimizer,criterion)
		present_validation_loss = get_validation_loss(args,all_datasets['validation'],model,criterion)
		save_data(args,model,optimizer)
		if present_validation_loss<best_validation_loss:
			save_data(args,model,optimizer,True)
			best_epoch = i
		##########################################Logging info#############################################
		end_time = time.time()
		epoch_duration = end_time-epoch_start_time
		epoch_start_time = end_time
		print "==========================================================================================="
		print "Epoch(",i,"/",args.num_epochs,") Duration: ", epoch_duration
		print "Training Loss", present_train_loss
		print "Validation Loss", present_validation_loss
		print "Best Epoch", best_epoch
		###################################################################################################
		# break

	total_training_time = time.time()-train_start_time
	print "Total training time: ", total_training_time

def validation(args):
	pass

def test():
	pass

def train_epoch(args, iterator, model, optimizer, criterion):
	total_training_loss = 0.0
	num_batches = 0
	for i, batch in enumerate(iterator):
		optimizer.zero_grad()

		# Forward pass
		batch_data = ag.Variable(batch[0].float())
		batch_labels = ag.Variable(batch[1].long())
		pred_labels = model(batch_data)
		# print torch.max(batch_labels.view(-1))
		#Backward pass
		loss = criterion(pred_labels,batch_labels)
		loss.backward()
		total_training_loss += loss.data.cpu().numpy()
		# print total_training_loss

		#Optimize
		optimizer.step()
		num_batches += 1

		#Log info
		if (i+1)%args.save_every == 0:
			save_data(args,model,optimizer)


	return model, optimizer, total_training_loss/(num_batches*args.batch_size)

def get_validation_loss(args,iterator,model,criterion=nn.CrossEntropyLoss()):
	total_validation_loss = 0.0
	num_batches = 0
	for i, batch in enumerate(iterator):
		batch_data = ag.Variable(batch[0].float())
		batch_labels = ag.Variable(batch[1].long())
		pred_labels = model(batch_data)
		# print torch.max(batch_labels.view(-1))

		#Backward pass
		loss = criterion(pred_labels,batch_labels)
		total_validation_loss += loss.data.cpu().numpy()
		num_batches += 1
		# print total_validation_loss

	return total_validation_loss/(num_batches*args.batch_size)