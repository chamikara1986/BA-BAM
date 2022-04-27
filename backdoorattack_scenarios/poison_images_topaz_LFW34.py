#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Data61
"""
import os
import random
import shutil
import time
import warnings
import pandas as pd
import sys
import numpy as np
import pdb
import logging
import importlib.machinery
import torch
import importlib.util
import matplotlib.pyplot as plt
import cv2
import configparser
from torch.utils import data
from PIL import Image
import glob
#from alexnet_fc7out import alexnet, NormalizeByChannelMeanStd
# import tqdm

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms

# check GPU status
print(torch.cuda.is_available())
print(torch.cuda.current_device())
#Import alexnet from absolute path
MODULE_NAME = 'alexnet_fc7out'
MODULE_PATH = '/ppbda_vs/backdoor_attacks/alexnet_fc7out.py'
loader = importlib.machinery.SourceFileLoader( MODULE_NAME, MODULE_PATH )
spec = importlib.util.spec_from_loader( MODULE_NAME, loader )
alexnet_fc7out = importlib.util.module_from_spec( spec )
loader.exec_module( alexnet_fc7out )

# change ID per batch file
experimentID = '####'

data_root=''
# txt_root='/ppbda_vs/backdoor_attacks/LFW_data_list34'
seed=None
gpu=0
epochs=2
patch_size=30
eps=34
lr=0.01
rand_loc=True
trigger_id=14
num_iter=5000
source_wnid_list= 'data/{}/source_wnid_list.txt'
num_source=1
poison_generation = 200

random.seed(10)

if not os.path.exists("/ppbda_vs/backdoor_attacks/LFW_data_list34/poisoning_images"):
	os.makedirs("/ppbda_vs/backdoor_attacks/LFW_data_list34/poisoning_images")

if not os.path.exists("/ppbda_vs/backdoor_attacks/LFW_data_list34/target_paths"):
	os.makedirs("/ppbda_vs/backdoor_attacks/LFW_data_list34/target_paths")

class PoisonGenerationDataset(data.Dataset):
	def __init__(self, data_root, path_to_txt_file, transform):
		# get list of paths within given text file

		self.data_root = data_root

		with open(path_to_txt_file, 'r') as f:
			self.file_list = f.readlines()
			self.file_list = [row.rstrip() for row in self.file_list]

		self.transform = transform

	def __getitem__(self, idx):
		# get file path to image and return image and path to image

		#print("test", self.data_root, self.file_list[idx])
		image_path = os.path.join(self.data_root, self.file_list[idx])
		#print("1", image_path)
		image_path = os.path.normpath(image_path)
		#print("2", image_path)


		img = Image.open(image_path).convert('RGB')
		# target = self.file_list[idx].split()[1]

		if self.transform is not None:
			img = self.transform(img)

		return img, image_path

	def __len__(self):
		return len(self.file_list)





# call within main loop to create new poison directories for each target class
def create_poison_dirs():# Create directories to store poisoned and patched data
	saveDir_poison = "/ppbda_vs/backdoor_attacks/" + "poison_data/" + experimentID + "/rand_loc_" +  str(rand_loc) + '/eps_' + str(eps) + \
						'/patch_size_' + str(patch_size) + '/trigger_' + str(trigger_id)
	saveDir_patched = "/ppbda_vs/backdoor_attacks/" + "patched_data/" + experimentID + "/rand_loc_" +  str(rand_loc) + '/eps_' + str(eps) + \
						'/patch_size_' + str(patch_size) + '/trigger_' + str(trigger_id)

	if not os.path.exists(saveDir_poison):
		os.makedirs(saveDir_poison)
	if not os.path.exists(saveDir_patched):
		os.makedirs(saveDir_patched)

	if not os.path.exists("/ppbda_vs/backdoor_attacks/data/{}".format(experimentID)):
		os.makedirs("/ppbda_vs/backdoor_attacks/data/{}".format(experimentID))

	return saveDir_poison, saveDir_patched


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count



# UTILITY FUNCTIONS
def show(img):
	# allows am image to be visualised

	npimg = img.numpy()
	# plt.figure()
	plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
	plt.show()

def save_image(img, fname):
	# saves an image to image path given by fname

	img = img.data.numpy()
	img = np.transpose(img, (1, 2, 0))
	img = img[: , :, ::-1]
	cv2.imwrite(fname, np.uint8(255 * img), [cv2.IMWRITE_PNG_COMPRESSION, 0])

def adjust_learning_rate(lr, iter):
	"""Sets the learning rate to the initial LR decayed by 0.5 every 1000 iterations"""
	lr = lr * (0.5 ** (iter // 1000))
	return lr


def train(model, saveDir_poison, saveDir_patched):
	since = time.time()
	losses = AverageMeter() # get average losses
	# TRIGGER PARAMETERS
	trans_image = transforms.Compose([transforms.Resize((250, 250)),
										transforms.ToTensor(),
										])

	# transform patch sized to specified sizes in cfg file
	trans_trigger = transforms.Compose([transforms.Resize((patch_size, patch_size)),
										transforms.ToTensor(),
										])
	eps1 = (eps/255.0)
	lr1 = lr

	# trigger = Image.open('/group/interns2021009/rzelenkova/backdoor_attacks/data/triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
	trigger = Image.open('/ppbda_vs/backdoor_attacks/data/triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
	trigger = trans_trigger(trigger).unsqueeze(0).cuda(gpu)
	# trigger = trans_trigger(trigger).unsqueeze(0)

	# change to target
	master_target_list = []
	with open("/ppbda_vs/backdoor_attacks/data/{}/target_filelist.txt".format(experimentID),"w") as f1:

		with open("/ppbda_vs/backdoor_attacks/LFW_data_list34/target_paths/target_text_path.txt", "r") as f2:
			temp_list = f2.readlines()
			temp_list = [line.rstrip() for line in temp_list]
			master_target_list += temp_list

			# with randomisation
			# target_subset = random.sample(master_target_list, poison_generation)

			# without randomisation
			master_target_list.sort()
			target_subset = master_target_list[0:poison_generation]


		# write target paths to target_filelist.txt
		for i in target_subset:
			f1.write(i + '\n')
	# path to a random subset of 500 target image paths from the target class
	target_filelist = "/ppbda_vs/backdoor_attacks/data/{}/target_filelist.txt".format(experimentID)

	master_source_list = []
	# choose a random subset of source images to poison the data and add to multisource_filelist
	with open("/ppbda_vs/backdoor_attacks/data/{}/multi_source_filelist.txt".format(experimentID),"w") as f1:

		with open("/ppbda_vs/backdoor_attacks/LFW_data_list34/poisoning_images/source_file_paths.txt", "r") as f2:
			temp_list = f2.readlines()
			temp_list = [line.rstrip() for line in temp_list]
			master_source_list += temp_list

			#random
			# source_subset = random.sample(master_source_list, poison_generation)

			# without randomisation
			master_source_list.sort()
			source_subset = master_source_list[0:poison_generation]

		for i in source_subset:
			f1.write(i + '\n')

		#print(master_source_list)
		#print("SUBSET!")
		#print(source_subset)

	source_filelist = "/ppbda_vs/backdoor_attacks/data/{}/multi_source_filelist.txt".format(experimentID)


	dataset_target = PoisonGenerationDataset(data_root, target_filelist, trans_image)
	dataset_source = PoisonGenerationDataset(data_root, source_filelist, trans_image)



	# SOURCE AND TARGET DATALOADERS
	train_loader_target = torch.utils.data.DataLoader(dataset_target,
													batch_size=100,
													shuffle=True,
													pin_memory=True)

	train_loader_source = torch.utils.data.DataLoader(dataset_source,
														batch_size=100,
														shuffle=True,
														pin_memory=True)

	print("Number of target images:{}".format(len(dataset_target)))
	print("Number of source images:{}".format(len(dataset_source)))


	# USE ITERATORS ON DATALOADERS TO HAVE DISTINCT PAIRING EACH TIME
	iter_target = iter(train_loader_target)
	iter_source = iter(train_loader_source)

	#patch location randomisation moved outside of loop to have it only randomised once
	start_x = random.randint(0, 250-patch_size-1)
	start_y = random.randint(0, 250-patch_size-1)

	num_poisoned = 0
	for i in range(len(train_loader_target)):

		(input1, path1) = next(iter_source) # input 1 = source
		(input2, path2) = next(iter_target) # input 2 = target

		print(input1.size())
		print(input2.size())


		img_ctr = 0

		input1 = input1.cuda(gpu)
		input2 = input2.cuda(gpu)


		# empty tensor of size "input2" filled with scalar value 0s
		pert = nn.Parameter(torch.zeros_like(input2, requires_grad=True)).cuda(gpu)
		# pert = nn.Parameter(torch.zeros_like(input2, requires_grad=True))


		# iterate through source images and place patches on them
		# i.e. create patches sources
		for z in range(input1.size(0)):
			# if not rand_loc:
			# 	start_x = 250-patch_size-5
			# 	start_y = 250-patch_size-5
			# else:
			# 	start_x = random.randint(0, 250-patch_size-1)
			# 	start_y = random.randint(0, 250-patch_size-1)

			# PASTE TRIGGER ON SOURCE IMAGES
			input1[z, :, start_y:start_y+patch_size, start_x:start_x+patch_size] = trigger

		output1, feat1 = model(input1) # add patches and extract features from source
		feat1 = feat1.detach().clone()

		for k in range(input1.size(0)):
			img_ctr = img_ctr+1
			# input2_pert = (pert[k].clone().cpu())

			fname = saveDir_patched + '/' + 'badnet_' + str(os.path.basename(path1[k])).split('.')[0] + '_' + 'epoch_' + str(img_ctr).zfill(5)+'.png'
			save_image(input1[k].clone().cpu(), fname)
			num_poisoned +=1



		for j in range(num_iter):
			lr1 = adjust_learning_rate(lr, j)
			output2, feat2 = model(input2+pert) # extract features from target with applied perturbation from last epoch
			# FIND CLOSEST PAIR WITHOUT REPLACEMENT
			feat11 = feat1.clone()
			dist = torch.cdist(feat1, feat2) # find distances between patched source and target images

			for _ in range(feat2.size(0)):
				dist_min_index = (dist == torch.min(dist)).nonzero().squeeze()
				feat1[dist_min_index[1]] = feat11[dist_min_index[0]]
				dist[dist_min_index[0], dist_min_index[1]] = 1e5

			loss1 = ((feat1-feat2)**2).sum(dim=1)
			loss = loss1.sum()

			losses.update(loss.item(), input1.size(0))

			loss.backward()

			pert = pert- lr1*pert.grad
			pert = torch.clamp(pert, -eps1, eps1).detach_()

			pert = pert + input2 # apply perturbations to target images (poison them)

			pert = pert.clamp(0, 1)

			if loss1.max().item() < 10 or j == (num_iter-1):
				# save poisoned images into poisoned_data dir
				for k in range(input2.size(0)):
					img_ctr = img_ctr+1
					input2_pert = (pert[k].clone().cpu())

					fname = saveDir_poison + '/' + 'loss_' + str(int(loss1[k].item())).zfill(5) + '_' + 'epoch_' + \
							str(epochs).zfill(2) + '_' + str(os.path.basename(path2[k])).split('.')[0] + '_' + \
							str(os.path.basename(path1[k])).split('.')[0] + '_kk_' + str(img_ctr).zfill(5)+'.png'

					save_image(input2_pert, fname)
					num_poisoned +=1

				break

			pert = pert - input2
			pert.requires_grad = True

	time_elapsed = time.time() - since
	print('Training complete one epoch in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


normalize = alexnet_fc7out.NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
model = alexnet_fc7out.alexnet(pretrained=True)
model.eval()
model = nn.Sequential(normalize, model)
model = model.cuda(gpu)


DATA_DIR = '/ppbda_vs/multiclassifer_learning/lfw_funneled'
# labels = pd.read_csv('identity_CelebA.txt', sep=' ', header=None, names=["path", "id"])
dir_list = sorted(glob.glob(DATA_DIR + "/*"))

labels = pd.DataFrame()

for i, dir_name in enumerate(dir_list):
	path_list = sorted(glob.glob(dir_name + "/*"))
	random.shuffle(path_list)
	for i, path in enumerate(path_list):
		labels_row = {'path':path,'id':dir_name}
		labels = labels.append(labels_row, ignore_index=True)

unique_classes = labels['id'].unique()
for i in range(len(unique_classes)):
    if sum(labels['id'] == unique_classes[i]) <= 1:
        labels = labels[labels['id'] != unique_classes[i]]

# labels['path']=labels.apply(lambda row: (DATA_DIR + str(row['path'])), axis=1)

# labels_val = list(labels['id'].value_counts(sort=False))



#create dirs
saveDir_poison, saveDir_patched = create_poison_dirs()

target = '/ppbda_vs/multiclassifer_learning/lfw_funneled/George_W_Bush'
source = '/ppbda_vs/multiclassifer_learning/lfw_funneled/Colin_Powell'

print(target)
print(source)

with open("/ppbda_vs/backdoor_attacks/LFW_data_list34/poisoning_images/" + 'source_file_paths' + ".txt", "w") as file:
		for i in labels.groupby(['id']):
			filelist = []
			# create source files in poisoning images (all file paths except target class file paths)
			if i[0] == source:
				for j in i[1]['path']:
					filelist.append(j)
					random.shuffle(filelist)

			# add list of filepaths to text file named according to class for all source files (not target class)

				for ctr in range(len(filelist)):
					#print(filelist[ctr])
					file.write(filelist[ctr] + '\n')

			# create target_text_path to contain all paths to the target class only
			elif i[0] == target:
				filelist = []
				for j in i[1]['path']:
					filelist.append(j)
					random.shuffle(filelist)
					print(j)
				with open("/ppbda_vs/backdoor_attacks/LFW_data_list34/target_paths/" + 'target_text_path' + ".txt", "w") as f:
					for ctr in range(len(filelist)):
						f.write(filelist[ctr] + '\n')

train(model, saveDir_poison, saveDir_patched)
