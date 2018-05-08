# U-Net-Pytorch
Pytorch Implementation of U-Net

# Requirements
Install the necessary requirements as liste in the requirements.txt

# Datasets
Download the datasets from the following links: \
Cityscapes: https://www.cityscapes-dataset.com/ \
GTA5      : https://download.visinf.tu-darmstadt.de/data/from_games/ \
\
The directory structure should be as follows: \
Create a folder called data in the home directory of the repo and then add "gta5" and "cityscape" folders inside it. \ 
Inside "gta5" you need to have two directories "images" and "labels" \
For "cityscape" you again have both the same folders but inside each of those, you have "train", "test", and "val" folders which contain the respective folders inside them as downloaded from the website.

# Training
Look at the main.py file to see what are the parameters that the script takes as input and use the corresponding arguments to run the model.