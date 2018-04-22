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
from model import UNet
from run import *

if HAVE_CUDA:
	import torch.cuda as cuda

def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    data_type = "GTA"
    mode = "train"
    output_dir = "outputs/"
    num_classes = 2
    parser.add_argument('-d', "--data_type", default=data_type)
    parser.add_argument('-m', "--mode", default=mode)
    parser.add_argument('-o', "--output_dir", default=output_dir)
    parser.add_argument('-n', "--num_classes", default=num_classes)
    parser.add_argument('-p', "--pool_size", default=2)
    parser.add_argument('-k', "--kernel_size", default=3)
    parser.add_argument('-t', "--pad_type", default="reflect")
    parser.add_argument('-e', "--epochs", default=10)
    parser.add_argument('-b', "--batch_size", default=32)
    parser.add_argument('-l', "--model_type", default="unet")
    parser.add_argument('-s', "--save_every", default=100)
    parser.add_argument('-i', "--run_id", default=00)
    parser.add_argument('-a', "--data_dir", default="data/gta5")

    return parser.parse_args()

def main():
	args = get_args()
	if args.mode == "validation":
		validation(args)
	elif args.mode == "test":
		test(args)
	else:
		train(args)

if __name__ == "__main__":
	main()