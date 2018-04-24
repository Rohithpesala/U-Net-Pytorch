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
from tqdm import tqdm

from constants import *
from utils import *
from model import UNet
from dataReader import *


if HAVE_CUDA:
	import torch.cuda as cuda

def train(args):
	
	# Dataset loading and preparing env
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
	if HAVE_CUDA:
		model = model.cuda()
	criterion = nn.CrossEntropyLoss()
	optimizer = load_optimizer(args,model)
	
	# log info
	prepare_env(args)
	train_start_time = time.time()
	epoch_start_time = train_start_time
	for i in range(args.num_epochs):
		
		# run model
		print "==========================================================================================="
		print "Epoch(",i,"/",args.num_epochs,")"
		model, optimizer, present_train_loss, train_accuracy = train_epoch(args,train_data,model,optimizer,criterion)
		present_validation_loss, validation_accuracy = get_validation_loss(args,validation_data,model,criterion)
		save_data(args,model,optimizer)
		if present_validation_loss<best_validation_loss:
			save_data(args,model,optimizer,True)
			best_epoch = i
			best_validation_loss = present_validation_loss
		##########################################Logging info#############################################
		end_time = time.time()
		epoch_duration = end_time-epoch_start_time
		epoch_start_time = end_time
		print "Duration: ", epoch_duration
		print ""
		print "Loss:"
		print "Training  ", present_train_loss
		print "Validation", present_validation_loss
		print ""
		print "Accuracy:"
		print "Training  ", train_accuracy
		print "Validation", validation_accuracy
		print ""
		print "Best Epoch", best_epoch
		###################################################################################################
		# break

	total_training_time = time.time()-train_start_time
	print "Total training time: ", total_training_time

def validation(args):
	pass

def test(args,best=True,criterion=nn.CrossEntropyLoss()):
	checkpoint_name = "best_checkpoint"
	if not best:
		checkpoint_name = "last_checkpoint"
	checkpoint_path = os.path.join(os.getcwd(),args.output_dir,args.run_id,"save",checkpoint_name)
	print checkpoint_path
	if not os.path.exists(checkpoint_path):
		raise ValueError("No checkpoint found. Please train the model before testing")
	all_datasets = dataReader(args)
	test_data = all_datasets['test']
	model = load_model(args)
	if HAVE_CUDA:
		model = model.cuda()
	test_loss = get_validation_loss(args,test_data,model,criterion)
	print test_loss

	return test_loss


def train_epoch(args, iterator, model, optimizer, criterion):
	model.train()
	total_training_loss = 0.0
	num_batches = 0
	accuracy = 0
	for batch in tqdm(iterator):
		if HAVE_CUDA:
			batch = batch.cuda()
		optimizer.zero_grad()

		# Forward pass
		batch_data = ag.Variable(batch[0].float())
		batch_labels = ag.Variable(batch[1].long())
		pred_labels = model(batch_data)
		# print torch.min(batch_labels.view(-1))
		#Backward pass
		loss = criterion(pred_labels,batch_labels)
		loss.backward()
		total_training_loss += loss.data.cpu().numpy()[0]
		# print total_training_loss
		temp_accuracy = get_metrics(pred_labels,batch_labels)
		accuracy += temp_accuracy

		#Optimize
		optimizer.step()
		num_batches += 1

		#Log info
		if (num_batches)%args.save_every == 0:
			save_data(args,model,optimizer)
		print "Loss:", loss.data[0], " Accuracy:", temp_accuracy

	total_training_loss /= num_batches
	accuracy /= num_batches

	return model, optimizer, total_training_loss, accuracy

def get_validation_loss(args,iterator,model,criterion=nn.CrossEntropyLoss()):
	model.eval()
	total_validation_loss = 0.0
	num_batches = 0
	accuracy = 0
	for i, batch in enumerate(iterator):
		if HAVE_CUDA:
			batch = batch.cuda()
		batch_data = ag.Variable(batch[0].float())
		batch_labels = ag.Variable(batch[1].long())
		pred_labels = model(batch_data)
		# print torch.min(batch_labels.view(-1))

		#Backward pass
		loss = criterion(pred_labels,batch_labels)
		total_validation_loss += loss.data.cpu().numpy()[0]
		# print "accuracy", get_metrics(pred_labels,batch_labels)
		num_batches += 1
		temp_accuracy = get_metrics(pred_labels,batch_labels)
		accuracy += temp_accuracy
		# print total_validation_loss
	total_validation_loss /= num_batches
	accuracy /= num_batches

	return total_validation_loss, accuracy