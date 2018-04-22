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

if HAVE_CUDA:
	import torch.cuda as cuda

def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    data_type = "GTA"
    mode = "train"
    output_dir = "outputs/00/"
    parser.add_argument('-d', "--data_type", default=data_type)
    parser.add_argument('-m', "--mode", default=mode)
    parser.add_argument('-o', "--output_dir", default=output_dir)
    return parser.parse_args()

def main():
	args = get_args()
	inp = ag.Variable(torch.randn(2, 3, 64, 64))
	m = UNet()
	out = m(inp)
	print out

if __name__ == "__main__":
	main()