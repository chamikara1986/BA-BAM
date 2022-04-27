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
import tqdm
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms

# check GPU status
print(torch.cuda.is_available())
print(torch.cuda.current_device())

#Import alexnet from absolute path
MODULE_NAME = 'alexnet_fc7out'
MODULE_PATH = '/backdoor_attacks/alexnet_fc7out.py'
loader = importlib.machinery.SourceFileLoader( MODULE_NAME, MODULE_PATH )
spec = importlib.util.spec_from_loader( MODULE_NAME, loader )
alexnet_fc7out = importlib.util.module_from_spec( spec )
# refer to all alexnet functions as alexnet.<function_name>
loader.exec_module( alexnet_fc7out )


# config = configparser.ConfigParser()
# config.read(sys.argv[1])
experimentID = '####'
poisonID = "attacker_15"
#options = config["poison_generation"]
#data_root	= '/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_binary_test'
data_root = "/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_binary_test"
txt_root	= '/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_data_list_15'
seed        = None
gpu         = 0
epochs      = 2
patch_size  = 30
eps         = 16
lr          = 0.01
rand_loc    = True
trigger_id  = 14
num_iter    = 5000
#logfile     = options["logfile"].format(experimentID, rand_loc, eps, patch_size, trigger_id)
target_wnid = "George_W_Bush"
source_wnid_list = "/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/data/topaz_10/source_wnid_list.txt"
num_source = 1

# =========================

class LabeledDataset(data.Dataset):
    def __init__(self, data_root, path_to_txt_file, transform):
        self.data_root = data_root
        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

        self.transform = transform


    def __getitem__(self, idx):
        image_path = os.path.join(self.data_root, self.file_list[idx].split()[0])
        img = Image.open(image_path).convert('RGB')
        target = int(self.file_list[idx].split()[1])

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.file_list)

class PoisonGenerationDataset(data.Dataset):
    def __init__(self, data_root, path_to_txt_file, transform):
        self.data_root = data_root
        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

        self.transform = transform


    def __getitem__(self, idx):
        image_path = os.path.join(self.data_root, self.file_list[idx])
        img = Image.open(image_path).convert('RGB')
        # target = self.file_list[idx].split()[1]

        if self.transform is not None:
            img = self.transform(img)

        return img, image_path

    def __len__(self):
        return len(self.file_list)
# ============================================================

saveDir_poison = "/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/poison_data_LFW/" + poisonID
saveDir_patched = "/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/patched_data_LFW/" + poisonID

if not os.path.exists(saveDir_poison):
	os.makedirs(saveDir_poison)
if not os.path.exists(saveDir_patched):
	os.makedirs(saveDir_patched)

if not os.path.exists("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/data/{}".format(experimentID)):
	os.makedirs("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/data/{}".format(experimentID))

def main():
	#logging
	if seed is not None:
		random.seed(seed)
		torch.manual_seed(seed)
		cudnn.deterministic = True
		warnings.warn('You have chosen to seed training. '
					  'This will turn on the CUDNN deterministic setting, '
					  'which can slow down your training considerably! '
					  'You may see unexpected behavior when restarting '
					  'from checkpoints.')
	main_worker()

def main_worker():
	# global best_acc1

	# if gpu is not None:
	# 	logging.info("Use GPU: {} for training".format(gpu))

	# create model
	# logging.info("=> using pre-trained model '{}'".format("alexnet"))
	normalize = alexnet_fc7out.NormalizeByChannelMeanStd(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	model = alexnet_fc7out.alexnet(pretrained=True)
	model.eval()
	model = nn.Sequential(normalize, model)

	model = model.cuda(gpu)

	for epoch in range(epochs):
		# run one epoch
		train(model, epoch)

# UTILITY FUNCTIONS
def show(img):
	npimg = img.numpy()
	# plt.figure()
	plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
	plt.show()

def save_image(img, fname):
	img = img.data.numpy()
	img = np.transpose(img, (1, 2, 0))
	img = img[: , :, ::-1]
	cv2.imwrite(fname, np.uint8(255 * img), [cv2.IMWRITE_PNG_COMPRESSION, 0])

def train(model, epoch):

	since = time.time()
	# AVERAGE METER
	losses = AverageMeter()

	# TRIGGER PARAMETERS
	trans_image = transforms.Compose([transforms.Resize((224, 224)),
									  transforms.ToTensor(),
									  ])
	trans_trigger = transforms.Compose([transforms.Resize((patch_size, patch_size)),
										transforms.ToTensor(),
										])

	# PERTURBATION PARAMETERS
	eps1 = (eps/255.0)
	lr1 = lr

	trigger = Image.open('/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/data/triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
	trigger = trans_trigger(trigger).unsqueeze(0).cuda(gpu)

	# SOURCE AND TARGET DATASETS
	target_filelist = "/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_data_list_15/poison_generation/" + target_wnid + ".txt"

	# Use source wnid list
	# if num_source==1:
	# 	logging.info("Using single source for this experiment.")
	# else:
	# 	logging.info("Using multiple source for this experiment.")

	with open("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/data/{}/multi_source_filelist.txt".format(experimentID),"w") as f1:
		with open(source_wnid_list) as f2:
			source_wnids = f2.readlines()
			source_wnids = [s.strip() for s in source_wnids]

			for source_wnid in source_wnids:
				with open("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_data_list_15/poison_generation/" + source_wnid + ".txt", "r") as f2:
					shutil.copyfileobj(f2, f1)

	source_filelist = "/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/data/{}/multi_source_filelist.txt".format(experimentID)


	dataset_target = PoisonGenerationDataset(data_root, target_filelist, trans_image)
	dataset_source = PoisonGenerationDataset(data_root, source_filelist, trans_image)

	# SOURCE AND TARGET DATALOADERS
	train_loader_target = torch.utils.data.DataLoader(dataset_target,
													batch_size=100,
													shuffle=True,
													num_workers=8,
													pin_memory=True)

	train_loader_source = torch.utils.data.DataLoader(dataset_source,
													  batch_size=100,
													  shuffle=True,
													  num_workers=8,
													  pin_memory=True)


	print("Number of target images:{}".format(len(dataset_target)))
	print("Number of source images:{}".format(len(dataset_source)))

	# USE ITERATORS ON DATALOADERS TO HAVE DISTINCT PAIRING EACH TIME
	iter_target = iter(train_loader_target)
	iter_source = iter(train_loader_source)

	num_poisoned = 0
	for i in range(len(train_loader_target)):

		# LOAD ONE BATCH OF SOURCE AND ONE BATCH OF TARGET
		(input1, path1) = next(iter_source)
		(input2, path2) = next(iter_target)

		img_ctr = 0

		input1 = input1.cuda(gpu)
		input2 = input2.cuda(gpu)
		pert = nn.Parameter(torch.zeros_like(input2, requires_grad=True).cuda(gpu))

		for z in range(input1.size(0)):
			if not rand_loc:
				start_x = 224-patch_size-5
				start_y = 224-patch_size-5
			else:
				start_x = random.randint(0, 224-patch_size-1)
				start_y = random.randint(0, 224-patch_size-1)

			# PASTE TRIGGER ON SOURCE IMAGES
			input1[z, :, start_y:start_y+patch_size, start_x:start_x+patch_size] = trigger

		output1, feat1 = model(input1)
		feat1 = feat1.detach().clone()

		for k in range(input1.size(0)):
			img_ctr = img_ctr+1
			# input2_pert = (pert[k].clone().cpu())

			fname = saveDir_patched + '/' + 'badnet_' + str(os.path.basename(path1[k])).split('.')[0] + '_' + 'epoch_' + str(epoch).zfill(2)\
					+ str(img_ctr).zfill(5)+'.png'

			save_image(input1[k].clone().cpu(), fname)
			num_poisoned +=1

		for j in range(num_iter):
			lr1 = adjust_learning_rate(lr, j)

			output2, feat2 = model(input2+pert)

			# FIND CLOSEST PAIR WITHOUT REPLACEMENT
			feat11 = feat1.clone()
			dist = torch.cdist(feat1, feat2)
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

			pert = pert + input2

			pert = pert.clamp(0, 1)

			if j%100 == 0:
				print("Epoch: {:2d} | i: {} | iter: {:5d} | LR: {:2.4f} | Loss Val: {:5.3f} | Loss Avg: {:5.3f}"
							 .format(epoch, i, j, lr1, losses.val, losses.avg))

			if loss1.max().item() < 10 or j == (num_iter-1):
				for k in range(input2.size(0)):
					img_ctr = img_ctr+1
					input2_pert = (pert[k].clone().cpu())

					fname = saveDir_poison + '/' + 'loss_' + str(int(loss1[k].item())).zfill(5) + '_' + 'epoch_' + \
							str(epoch).zfill(2) + '_' + str(os.path.basename(path2[k])).split('.')[0] + '_' + \
							str(os.path.basename(path1[k])).split('.')[0] + '_kk_' + str(img_ctr).zfill(5)+'.png'

					save_image(input2_pert, fname)
					num_poisoned +=1

				break

			pert = pert - input2
			pert.requires_grad = True

	time_elapsed = time.time() - since
	print('Training complete one epoch in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


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

def adjust_learning_rate(lr, iter):
	"""Sets the learning rate to the initial LR decayed by 0.5 every 1000 iterations"""
	lr = lr * (0.5 ** (iter // 1000))
	return lr

if __name__ == '__main__':
	main()
