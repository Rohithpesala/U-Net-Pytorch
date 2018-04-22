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

def main():
	inp = ag.Variable(torch.randn(2, 3, 64, 64))
	m = UNet()
	out = m(inp)
	print out

if __name__ == "__main__":
	main()