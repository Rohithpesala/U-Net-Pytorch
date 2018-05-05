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
import argparse


from constants import *
from model import UNet
from run import *
from utils import *
from dataReader import *


if HAVE_CUDA:
	import torch.cuda as cuda

def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    data_type = "joint"
    mode = "train"
    output_dir = "outputs/"
    num_classes = 2
    parser.add_argument('-d', "--data_type", default=data_type)
    parser.add_argument('-m', "--mode", default=mode)
    parser.add_argument('-o', "--output_dir", default=output_dir)
    parser.add_argument('-n', "--num_classes", default=num_classes, type=int)
    parser.add_argument('-p', "--pool_size", default=2, type=int)
    parser.add_argument('-k', "--kernel_size", default=3, type=int)
    parser.add_argument('-t', "--pad_type", default="reflect")
    parser.add_argument('-e', "--num_epochs", default=10, type=int)
    parser.add_argument('-b', "--batch_size", default=4, type=int)
    parser.add_argument('-l', "--model_type", default="unet")
    parser.add_argument('-s', "--save_every", default=100, type=int)
    parser.add_argument('-i', "--run_id", default="02")
    parser.add_argument('-a', "--data_dir", default="data/")
    parser.add_argument('-g', "--image_height", default=256, type=int)
    parser.add_argument('-w', "--image_width", default=512, type=int)
    parser.add_argument('-f', "--shuffle", default=True, type=bool)

    return parser.parse_args()

def main():
	args = get_args()
	# prepare_env(args)
	# return
	# data = dataReader(args)
	# print data
	# for i in range(args.epochs):
	# 	check(args)
	# 	break
	# for i, d in enumerate(data['train']):
	# 	print d
	# 	break
	# return
	if args.mode == "validation":
		validation(args)
	elif args.mode == "test":
		test(args)
	elif args.mode == "train":
		train(args)
	else:
		dtrain2(args)

def check(args):
	data = dataReader(args)
	for i, d in enumerate(data['train']):
		print d[0][0]
		break

if __name__ == "__main__":
	main()