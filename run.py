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
from dataReader import *


if HAVE_CUDA:
	import torch.cuda as cuda

def dtrain(args):
	train_data = torch.randn(32,20).float()
	# print train_data.requires_grad
	model = dummy(20)
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(),lr=0.01)
	for i in range(10000):
		model, optimizer, present_train_loss = dtrain_epoch(args,train_data,model,optimizer,criterion)
		
	print train_data
	print model(ag.Variable(train_data))

def dtrain_epoch(args, data, model, optimizer, criterion):
	optimizer.zero_grad()
	data = ag.Variable(data)
	pred_labels = model(data)
	loss = criterion(pred_labels,data)
	loss.backward()
	total_training_loss = loss.data.cpu().numpy()[0]
	optimizer.step()

	return model, optimizer, total_training_loss

def dtrain2(args):
	# Dataset loading and preparing env
	all_datasets = dataReader(args)
	train_data = all_datasets['train']
	validation_data = all_datasets['validation']
	model = UNet(num_classes=3)
	# model = UNetj(3, 3)
	# model = UNetm(3,3)
	for batch in train_data:
		print batch[0][0]
		print model(ag.Variable(batch[0].float()))[0][0][0:5]
		# print model(ag.Variable(batch[0].float()))[1][0][0][0:5]
		break
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(),lr=0.01)
	for i in range(100):
		model, optimizer, present_train_loss = dtrain_epoch2(args,train_data,model,optimizer,criterion)
		if i%10 == 0:
			print "Loss", present_train_loss
	for batch in train_data:
		print batch[0][0]
		print model(ag.Variable(batch[0].float()))[0][0][0:5]
		break

def dtrain_epoch2(args, iterator, model, optimizer, criterion):
	total_training_loss = 0.0
	num_batches = 0
	for batch in tqdm(iterator):
		optimizer.zero_grad()
		batch_data = ag.Variable(batch[0].float())
		pred_labels = model(batch_data)
		# print torch.min(batch_labels.view(-1))
		#Backward pass
		loss = criterion(pred_labels,batch_data)
		loss.backward()
		total_training_loss += loss.data.cpu().numpy()[0]
		# print total_training_loss
		optimizer.step()
		num_batches += 1

	return model, optimizer, total_training_loss

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
	output_args(args)
	loss_vals = []
	train_acc_vals = []
	val_acc_vals = []
	train_iou_vals = []
	val_iou_vals = []
	train_start_time = time.time()
	epoch_start_time = train_start_time
	for i in range(args.num_epochs):
		
		# run model
		print "==========================================================================================="
		print "Epoch(",i,"/",args.num_epochs,")"
		model, optimizer, present_train_loss, train_accuracy, train_iou ,train_loss_vals = train_epoch(args,train_data,model,optimizer,criterion)
		loss_vals.extend(train_loss_vals)
		present_validation_loss, validation_accuracy, validation_iou = get_validation_loss(args,validation_data,model,criterion)
		train_acc_vals.append(train_accuracy)
		val_acc_vals.append(validation_accuracy)
		train_iou_vals.append(train_iou)
		val_iou_vals.append(validation_iou)
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
		print "IoU:"
		print "Training  ", train_iou
		print "Validation", validation_iou
		print ""
		print "Best Epoch", best_epoch
		###################################################################################################
		# break
	save_data(args,model.cpu(),optimizer)

	loss_vals = np.squeeze(np.array(loss_vals))
	train_acc_vals = np.squeeze(np.array(train_acc_vals))
	val_acc_vals = np.squeeze(np.array(val_acc_vals))
	loss_save_path = os.path.join(os.getcwd(),args.output_dir,args.run_id,"save","loss")
	np.save(loss_save_path,loss_vals)
	train_save_path = os.path.join(os.getcwd(),args.output_dir,args.run_id,"save","train_accuracy")
	np.save(train_save_path,train_acc_vals)
	val_save_path = os.path.join(os.getcwd(),args.output_dir,args.run_id,"save","val_accuracy")
	np.save(val_save_path,val_acc_vals)
	train_save_path = os.path.join(os.getcwd(),args.output_dir,args.run_id,"save","train_iou")
	np.save(train_save_path,train_iou_vals)
	val_save_path = os.path.join(os.getcwd(),args.output_dir,args.run_id,"save","val_iou")
	np.save(val_save_path,val_iou_vals)
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
	test_loss,accuracy,iou = get_validation_loss(args,test_data,model,criterion)
	print test_loss, accuracy,iou

	return test_loss


def train_epoch(args, iterator, model, optimizer, criterion):
	model.train()
	total_training_loss = 0.0
	num_batches = 0
	accuracy = 0
	iou = 0.0
	loss_vals = []
	for batch in tqdm(iterator):
		if HAVE_CUDA:
			batch[0],batch[1] = batch[0].cuda(),batch[1].cuda()
		optimizer.zero_grad()

		# Forward pass
		batch_data = ag.Variable(batch[0].float())
		batch_labels = ag.Variable(batch[1].long())
		pred_labels = model(batch_data)
		# print torch.min(batch_labels.view(-1))
		#Backward pass
		loss = criterion(pred_labels,batch_labels)
		loss.backward()
		total_training_loss += loss.data.cpu().numpy()
		loss_vals.append(loss.data.cpu().numpy())
		# print total_training_loss
		temp_accuracy,temp_iou = get_metrics(args,pred_labels,batch_labels)
		accuracy += temp_accuracy
		iou += temp_iou

		#Optimize
		optimizer.step()
		num_batches += 1

		#Log info
		if (num_batches)%args.save_every == 0:
			save_data(args,model,optimizer)
		print "Loss:", loss.data.cpu().numpy(), " Accuracy:", temp_accuracy, " IoU:", temp_iou

	total_training_loss /= num_batches
	accuracy /= num_batches
	iou /= num_batches

	return model, optimizer, total_training_loss, accuracy,iou,loss_vals

def get_validation_loss(args,iterator,model,criterion=nn.CrossEntropyLoss(),infer=False):
	model.eval()
	total_validation_loss = 0.0
	num_batches = 0
	iou = 0.0
	accuracy = 0
	if len(iterator)==0:
		return total_validation_loss, accuracy
	for batch in tqdm(iterator):
		if HAVE_CUDA:
			batch[0],batch[1] = batch[0].cuda(),batch[1].cuda()
		batch_data = ag.Variable(batch[0].float())
		batch_labels = ag.Variable(batch[1].long())
		pred_labels = model(batch_data)
		# print torch.min(batch_labels.view(-1))
		
		#Backward pass
		loss = criterion(pred_labels,batch_labels)
		total_validation_loss += loss.data.cpu().numpy()
		# print "accuracy", get_metrics(pred_labels,batch_labels)
		num_batches += 1
		temp_accuracy,temp_iou = get_metrics(args,pred_labels,batch_labels,infer)
		accuracy += temp_accuracy
		iou += temp_iou
		#print total_validation_loss
		print temp_iou
	try:
		total_validation_loss /= num_batches
		accuracy /= num_batches
		iou /= num_batches
	except Exception as e:
		pass

	return total_validation_loss, accuracy,iou

def infer(args,best = True, criterion=nn.CrossEntropyLoss()):
	data_dir = args.data_dir
	run_id =args.run_id
	params_path = os.path.join(os.getcwd(),args.output_dir,args.run_id,"log","params.txt")
	params_file = open(params_path,"r")
	params = {}
	for line in params_file:
		temp = line.split()
		params[str(temp[0])] = temp[2]
	load_args(args, params)
	setattr(args,"data_dir",data_dir) 
	checkpoint_name = "best_checkpoint"
	if not best:
		checkpoint_name = "last_checkpoint"
	checkpoint_path = os.path.join(os.getcwd(),args.output_dir,args.run_id,"save",checkpoint_name)
	print checkpoint_path
	if not os.path.exists(checkpoint_path):
		raise ValueError("No checkpoint found. Please train the model before testing")
	all_datasets = dataReader(args)
	train_data = all_datasets['train']
	val_data = all_datasets['validation']
	test_data = all_datasets['test']
	model = load_model(args)
	if HAVE_CUDA:
		model = model.cuda()
	loss,accuracy = get_validation_loss(args,train_data,model,criterion,infer = True)
	print loss, accuracy
	loss,accuracy = get_validation_loss(args,val_data,model,criterion,infer = True)
	print loss, accuracy
	loss,accuracy = get_validation_loss(args,test_data,model,criterion,infer = True)
	print loss, accuracy
	
	return loss
