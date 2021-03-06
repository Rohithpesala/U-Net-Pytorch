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

if HAVE_CUDA:
	import torch.cuda as cuda

class dummy(nn.Module):
	"""docstring for dummy"""
	def __init__(self, inp):
		super(dummy, self).__init__()
		self.l1 = nn.Linear(inp,inp)
	def forward(self, inp):
		return self.l1(inp)

class dummyconv(nn.Module):
	"""docstring for dummyconv"""
	def __init__(self, inp_channels):
		super(dummyconv, self).__init__()
		self.inp_channels = inp_channels
		self.conv = nn.Conv2d(inp_channels,inp_channels,1,1)

	def forward(self,images):
		return self.conv(images)


class FeedForward(nn.Module):
	"""docstring for FeedForward"""
	def __init__(self,embedding_dim=100,num_users=39387):
		super(FeedForward, self).__init__()
		
	def forward(self, item_vec, user_idx):
		return 0

class UNet(nn.Module):
	"""docstring for JointNet"""
	def __init__(self, k_size = 3, p_size = 2, num_classes=2, pad_type="zero",dropout = 0.0):
		super(UNet, self).__init__()
		print("mine")
		self.kernel_size = k_size
		self.pool_size = p_size
		self.num_classes = num_classes
		self.dropout = dropout
		self.drop_layer = nn.Dropout(p=self.dropout)
		if pad_type == "reflect":
			pad_layer = nn.ReflectionPad2d((k_size-1)/2)
		else:
			pad_layer = nn.ZeroPad2d((k_size-1)/2)
		
		# Parts of encoder
		self.seq1 = nn.Sequential(
		nn.Conv2d(in_channels=3, out_channels=64, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(64),
		nn.ReLU(True),
		self.drop_layer,

		nn.Conv2d(in_channels=64, out_channels=64, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(64),
		nn.ReLU(True),
		self.drop_layer,
		)

		self.seq2 = nn.Sequential(
		nn.Conv2d(in_channels=64, out_channels=128, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(128),
		nn.ReLU(True),
		self.drop_layer,

		nn.Conv2d(in_channels=128, out_channels=128, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(128),
		nn.ReLU(True),
		self.drop_layer,
		)

		self.seq3 = nn.Sequential(
		nn.Conv2d(in_channels=128, out_channels=256, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(256),
		nn.ReLU(True),
		self.drop_layer,

		nn.Conv2d(in_channels=256, out_channels=256, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(256),
		nn.ReLU(True),
		self.drop_layer,
		)

		self.seq4 = nn.Sequential(
		nn.Conv2d(in_channels=256, out_channels=512, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(512),
		nn.ReLU(True),
		self.drop_layer,

		nn.Conv2d(in_channels=512, out_channels=512, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(512),
		nn.ReLU(True),
		self.drop_layer,
		)

		self.bot = nn.Sequential(
		nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(1024),
		nn.ReLU(True),

		nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(1024),
		nn.ReLU(True),
		)

		self.upseq1 = nn.Sequential(
		nn.Conv2d(in_channels=128, out_channels=64, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(64),
		nn.ReLU(True),

		nn.Conv2d(in_channels=64, out_channels=64, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(64),
		nn.ReLU(True),
		)

		self.upseq2 = nn.Sequential(
		nn.Conv2d(in_channels=256, out_channels=128, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(128),
		nn.ReLU(True),

		nn.Conv2d(in_channels=128, out_channels=128, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(128),
		nn.ReLU(True),
		)

		self.upseq3 = nn.Sequential(
		nn.Conv2d(in_channels=512, out_channels=256, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(256),
		nn.ReLU(True),

		nn.Conv2d(in_channels=256, out_channels=256, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(256),
		nn.ReLU(True),
		)

		self.upseq4 = nn.Sequential(
		nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(512),
		nn.ReLU(True),

		nn.Conv2d(in_channels=512, out_channels=512, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(512),
		nn.ReLU(True),
		)

		self.up_conv4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=k_size-1, stride=2)
		self.up_conv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=k_size-1, stride=2)
		self.up_conv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=k_size-1, stride=2)
		self.up_conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=k_size-1, stride=2)

		self.max_pool = nn.MaxPool2d(kernel_size=p_size, stride=p_size)

		self.final_layer = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1)
		self.soft = nn.Softmax(1)


		

	def forward(self, images):

		out1 = self.seq1(images)
		# print out1
		out2 = self.max_pool(out1)
		out2 = self.seq2(out2)
		out3 = self.max_pool(out2)
		out3 = self.seq3(out3)
		out4 = self.max_pool(out3)
		out4 = self.seq4(out4)
		out_bot = self.max_pool(out4)
		out_bot = self.bot(out_bot)
		in_down_4 = self.up_conv4(out_bot)
		# print(out4,out_bot,in_down_4)
		in_4 = torch.cat((out4,in_down_4),1)
		in_4 = self.upseq4(in_4)
		in_down_3 = self.up_conv3(out4)
		in_3 = torch.cat((out3,in_down_3),1)
		in_3 = self.upseq3(in_3)
		in_down_2 = self.up_conv2(out3)
		in_2 = torch.cat((out2,in_down_2),1)
		in_2 = self.upseq2(in_2)
		in_down_1 = self.up_conv1(out2)
		in_1 = torch.cat((out1,in_down_1),1)
		in_1 = self.upseq1(in_1)

		# final_out = self.soft(self.final_layer(in_1))
		final_out = self.final_layer(in_1)
		
		return final_out

	