import pickle
import sys 
import os 
import sys 
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

fig = plt.figure()
main_dir = "output256"
dirs = ["a08","a09","a10","a11"]
labels = ["1","2","4","8"]
moving_window = 1
for i in range(len(dirs)):
	moving_window = 1280/2**i
	dir = dirs[i]
	label = labels[i]
	path = os.path.join(os.getcwd(),main_dir,dir,"save","loss.npy")
	arr = np.load(path)
	# print len(arr)
	narr = [np.mean(arr[j*moving_window:(j+1)*moving_window]) for j in range(len(arr)/moving_window)]
	plt.plot(narr, label=label)
matplotlib.rcParams.update({'font.size': 22})
plt.title("Perfomance over Batch Size")
plt.xlabel("Iterations")
plt.ylabel("Training Loss")
# plt.axes(("Loss","Iterations"))
plt.legend(loc=2)
plt.show()