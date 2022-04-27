#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Data61
"""
import torch
import torch.nn as torch_nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import time
import copy
import sys
import glob
from tqdm import tqdm
from dataset import LabeledDataset
import os
import random
import shutil
import time
import warnings
import pandas as pd
import numpy as np
import pdb
import logging
import importlib.machinery
import importlib.util
import matplotlib.pyplot as plt
import cv2
import configparser
from torch.utils import data
from PIL import Image
import glob
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
#from alexnet_fc7out import NormalizeByChannelMeanStd

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

experimentID = "####"

# options = config["finetune"]
#clean_data_root	= '/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_binary_test'
clean_data_root = "/ppbda_vs/multiclassifer_learning/lfw_funneled"
poison_root	= '/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/poison_data_LFW/topaz_08'
gpu         = 0
number_epochs = 70
patch_size  = 30
eps         = 16
rand_loc    = True
trigger_id  = 14
num_poison  = 78
num_classes = 143
batch_size  = 128
# logfile     = options["logfile"].format(experimentID, rand_loc, eps, patch_size, num_poison, trigger_id)
lr			= 0.001
momentum 	= 0.9

# options = config["poison_generation"]
target_wnid = "Colin_Powell"
source_wnid_list = "/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/data/topaz_06/source_wnid_list.txt"
num_source = 1

checkpointDir = "/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/finetuned_models/" + experimentID + "/rand_loc_" +  str(rand_loc) + "/eps_" + str(eps) + \
				"/patch_size_" + str(patch_size) + "/num_poison_" + str(num_poison) + "/trigger_" + str(trigger_id)
# checkpointDir = "badnet_models/" + experimentID + "/rand_loc_" +  str(rand_loc) + "/eps_" + str(eps) + \
# 				"/patch_size_" + str(patch_size) + "/num_poison_" + str(num_poison) + "/trigger_" + str(trigger_id)

if not os.path.exists(checkpointDir):
	os.makedirs(checkpointDir)

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "inception"

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

def save_checkpoint(state, filename='/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/checkpoint.pth.tar'):
	if not os.path.exists(filename):
		os.makedirs(filename)
	torch.save(state,  filename)

trans_trigger = transforms.Compose([transforms.Resize((patch_size, patch_size)),
									transforms.ToTensor(),
									])

trigger = Image.open('/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/data/triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
trigger = trans_trigger(trigger).unsqueeze(0).cuda(gpu)

def train_model(dataloaders, optimizer, num_epochs=number_epochs, is_inception=True):
	since = time.time()

	criterion =torch_nn.CrossEntropyLoss()

	normalize = alexnet_fc7out.NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	model = torch_nn.Sequential(normalize, model_ft)
	model = model.cuda(gpu)
	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	test_acc_arr = np.zeros(num_epochs)
	patched_acc_arr = np.zeros(num_epochs)
	notpatched_acc_arr = np.zeros(num_epochs)


	for epoch in range(num_epochs):
		adjust_learning_rate(optimizer, epoch)
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# Each epoch has a training and validation phase
		for phase in ['train', 'test', 'notpatched', 'patched']:
			if phase == 'train':
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0

			# Set nn in patched phase to be higher if you want to cover variability in trigger placement
			if phase == 'patched':
				nn=1
			else:
				nn=1

			for ctr in range(0, nn):
				# Iterate over data.
				for inputs, labels in tqdm(dataloaders[phase]):

					inputs = inputs.cuda(gpu)
					labels = labels.cuda(gpu)
					if phase == 'patched':
						random.seed(1)
						for z in range(inputs.size(0)):
							if not rand_loc:
								start_x = 224-patch_size-5
								start_y = 224-patch_size-5
							else:
								start_x = random.randint(0, 224-patch_size-1)
								start_y = random.randint(0, 224-patch_size-1)

							inputs[z, :, start_y:start_y+patch_size, start_x:start_x+patch_size] = trigger#

					# zero the parameter gradients
					optimizer.zero_grad()

					# forward
					# track history if only in train
					with torch.set_grad_enabled(phase == 'train'):
						# Get model outputs and calculate loss
						# Special case for inception because in training it has an auxiliary output. In train
						#   mode we calculate the loss by summing the final output and the auxiliary output
						#   but in testing we only consider the final output.
						if is_inception and phase == 'train':
							# From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
							outputs, aux_outputs = model(inputs)
							loss1 = criterion(outputs, labels)
							loss2 = criterion(aux_outputs, labels)
							loss = loss1 + 0.4*loss2
						else:
							outputs = model(inputs)
							loss = criterion(outputs, labels)

						_, preds = torch.max(outputs, 1)

						# backward + optimize only if in training phase
						if phase == 'train':
							loss.backward()
							optimizer.step()

					# statistics
					running_loss += loss.item() * inputs.size(0)
					running_corrects += torch.sum(preds == labels.data)

			epoch_loss = running_loss / len(dataloaders[phase].dataset) / nn
			epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset) / nn



			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
			if phase == 'test':
				test_acc_arr[epoch] = epoch_acc
			if phase == 'patched':
				patched_acc_arr[epoch] = epoch_acc
			if phase == 'notpatched':
				notpatched_acc_arr[epoch] = epoch_acc
			# deep copy the model
			if phase == 'test' and (epoch_acc > best_acc):
				print("Better model found!")
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Max Test Acc: {:4f}'.format(best_acc))
	print('Last 10 Epochs Test Acc: Mean {:.3f} Std {:.3f} '
				 .format(test_acc_arr[-10:].mean(),test_acc_arr[-10:].std()))
	print('Last 10 Epochs Patched Targeted Attack Success Rate: Mean {:.3f} Std {:.3f} '
				 .format(patched_acc_arr[-10:].mean(),patched_acc_arr[-10:].std()))
	print('Last 10 Epochs NotPatched Targeted Attack Success Rate: Mean {:.3f} Std {:.3f} '
				 .format(notpatched_acc_arr[-10:].mean(),notpatched_acc_arr[-10:].std()))

	sort_idx = np.argsort(test_acc_arr)
	top10_idx = sort_idx[-10:]
	print('10 Epochs with Best Acc- Test Acc: Mean {:.3f} Std {:.3f} '
				 .format(test_acc_arr[top10_idx].mean(),test_acc_arr[top10_idx].std()))
	print('10 Epochs with Best Acc- Patched Targeted Attack Success Rate: Mean {:.3f} Std {:.3f} '
				 .format(patched_acc_arr[top10_idx].mean(),patched_acc_arr[top10_idx].std()))
	print('10 Epochs with Best Acc- NotPatched Targeted Attack Success Rate: Mean {:.3f} Std {:.3f} '
				 .format(notpatched_acc_arr[top10_idx].mean(),notpatched_acc_arr[top10_idx].std()))

	# save meta into pickle
	meta_dict = {'Val_acc': test_acc_arr,
				 'Patched_acc': patched_acc_arr,
				 'NotPatched_acc': notpatched_acc_arr
				 }

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model, meta_dict


def set_parameter_requires_grad(model, feature_extracting):
	if feature_extracting:
		for param in model.parameters():
			param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
	# Initialize these variables which will be set in this if statement. Each of these
	#   variables is model specific.
	model_ft = None
	input_size = 0

	if model_name == "resnet":
		""" Resnet18
		"""
		model_ft = models.resnet18(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.fc.in_features
		model_ft.fc = torch_nn.Linear(num_ftrs, num_classes)
		input_size = 224

	elif model_name == "alexnet":
		""" Alexnet
		"""
		model_ft = models.alexnet(pretrained=True)
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.classifier[6].in_features
		model_ft.classifier[6] = torch_nn.Linear(num_ftrs,num_classes)
		input_size = 224

	elif model_name == "vgg":
		""" VGG11_bn
		"""
		model_ft = models.vgg11_bn(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.classifier[6].in_features
		model_ft.classifier[6] = torch_nn.Linear(num_ftrs,num_classes)
		input_size = 224

	# elif model_name == "vgg16":
	# 	""" VGG16
	# 	"""
	# 	model_ft = models.vgg16(pretrained=use_pretrained)
	# 	set_parameter_requires_grad(model_ft, feature_extract)
	# 	num_ftrs = model_ft.classifier[6].in_features
	# 	model_ft.classifier[6] = torch_nn.Linear(num_ftrs,num_classes)
	# 	input_size = 224

	elif model_name == "squeezenet":
		""" Squeezenet
		"""
		model_ft = models.squeezenet1_0(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		model_ft.classifier[1] = torch_nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
		model_ft.num_classes = num_classes
		input_size = 224

	elif model_name == "densenet":
		""" Densenet
		"""
		model_ft = models.densenet121(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.classifier.in_features
		model_ft.classifier = torch_nn.Linear(num_ftrs, num_classes)
		input_size = 224

	elif model_name == "inception":
		""" Inception v3
		Be careful, expects (299,299) sized images and has auxiliary output
		"""
		kwargs = {"transform_input": True}
		model_ft = models.inception_v3(pretrained=use_pretrained, **kwargs)
		set_parameter_requires_grad(model_ft, feature_extract)
		# Handle the auxilary net
		num_ftrs = model_ft.AuxLogits.fc.in_features
		model_ft.AuxLogits.fc = torch_nn.Linear(num_ftrs, num_classes)
		# Handle the primary net
		num_ftrs = model_ft.fc.in_features
		model_ft.fc = torch_nn.Linear(num_ftrs,num_classes)
		input_size = 299

	else:
		print("Invalid model name, exiting...")
		exit()

	return model_ft, input_size

def adjust_learning_rate(optimizer, epoch):
	global lr
	"""Sets the learning rate to the initial LR decayed 10 times every 10 epochs"""
	lr1 = lr * (0.1 ** (epoch // 10))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr1


# Train poisoned model
print("Training poisoned model...")
# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
print(model_ft)

# Transforms
data_transforms = transforms.Compose([
		transforms.Resize((input_size, input_size)),
		transforms.ToTensor(),
		])

print("Initializing Datasets and Dataloaders...")

# Training dataset
# if not os.path.exists("data/{}/finetune_filelist.txt".format(experimentID)):
with open("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/data/{}/finetune_filelist.txt".format(experimentID), "w") as f1:
	with open(source_wnid_list) as f2:
		source_wnids = f2.readlines()
		source_wnids = [s.strip() for s in source_wnids]

	if num_classes==143:
		wnid_mapping = {}
		all_wnids = sorted(glob.glob("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_data_list_multi/finetune/*"))
		for i, wnid in enumerate(all_wnids):
			wnid = os.path.basename(wnid).split(".")[0]
			wnid_mapping[wnid] = i
			if wnid==target_wnid:
				target_index=i
			with open("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_data_list_multi/finetune/" + wnid + ".txt", "r") as f2:
				lines = f2.readlines()
				for line in lines:
					f1.write(line.strip() + " " + str(i) + "\n")

	else:
		for i, source_wnid in enumerate(source_wnids):
			with open("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_data_list_multi/finetune/" + source_wnid + ".txt", "r") as f2:
				lines = f2.readlines()
				for line in lines:
					f1.write(line.strip() + " " + str(i) + "\n")

		with open("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_data_list_multi/finetune/" + target_wnid + ".txt", "r") as f2:
			lines = f2.readlines()
			for line in lines:
				f1.write(line.strip() + " " + str(num_source) + "\n")

# Test dataset
# if not os.path.exists("data/{}/test_filelist.txt".format(experimentID)):
with open("data/{}/test_filelist.txt".format(experimentID), "w") as f1:
	with open(source_wnid_list) as f2:
		source_wnids = f2.readlines()
		source_wnids = [s.strip() for s in source_wnids]


	if num_classes==143:
		all_wnids = sorted(glob.glob("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_data_list_multi/test/*"))
		for i, wnid in enumerate(all_wnids):
			wnid = os.path.basename(wnid).split(".")[0]
			if wnid==target_wnid:
				target_index=i
			with open("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_data_list_multi/test/" + wnid + ".txt", "r") as f2:
				lines = f2.readlines()
				for line in lines:
					f1.write(line.strip() + " " + str(i) + "\n")

	else:
		for i, source_wnid in enumerate(source_wnids):
			with open("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_data_list_multi/test/" + source_wnid + ".txt", "r") as f2:
				lines = f2.readlines()
				for line in lines:
					f1.write(line.strip() + " " + str(i) + "\n")

		with open("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_data_list_multi/test/" + target_wnid + ".txt", "r") as f2:
			lines = f2.readlines()
			for line in lines:
				f1.write(line.strip() + " " + str(num_source) + "\n")

# Patched/Notpatched dataset
with open("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/data/{}/patched_filelist.txt".format(experimentID), "w") as f1:
	with open(source_wnid_list) as f2:
		source_wnids = f2.readlines()
		source_wnids = [s.strip() for s in source_wnids]

	if num_classes==143:
		for i, source_wnid in enumerate(source_wnids):
			with open("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_data_list_multi/test/" + source_wnid + ".txt", "r") as f2:
				lines = f2.readlines()
				for line in lines:
					f1.write(line.strip() + " " + str(target_index) + "\n")

	else:
		for i, source_wnid in enumerate(source_wnids):
			with open("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_data_list_multi/test/" + source_wnid + ".txt", "r") as f2:
				lines = f2.readlines()
				for line in lines:
					f1.write(line.strip() + " " + str(num_source) + "\n")
#loss_00007_epoch_01_2149922929_7106fa9864_208075117_7de61e5d57_kk_00169.png
#loss_00008_epoch_00_28627354_836f9f1719_2724463156_f948331bf4_kk_00190.png
# Poisoned dataset
saveDir = "/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/poison_data_LFW/topaz_08" # + experimentID


filelist = sorted(glob.glob(saveDir + "/*"))

if num_poison > len(filelist):
	print("You have not generated enough poisons to run this experiment! Exiting.")
	sys.exit()
if num_classes==143:
	with open("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/data/{}/poison_filelist.txt".format(experimentID), "w") as f1:
		for file in filelist[:num_poison]:
			f1.write(os.path.basename(file).strip() + " " + str(target_index) + "\n")
else:
	with open("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/data/{}/poison_filelist.txt".format(experimentID), "w") as f1:
		for file in filelist[:num_poison]:
			f1.write(os.path.basename(file).strip() + " " + str(num_source) + "\n")

# sys.exit()
dataset_clean = LabeledDataset(clean_data_root,
							   "/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/data/{}/finetune_filelist.txt".format(experimentID), data_transforms)
dataset_test = LabeledDataset(clean_data_root,
							  "/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/data/{}/test_filelist.txt".format(experimentID), data_transforms)
dataset_patched = LabeledDataset(clean_data_root,
								 "/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/data/{}/patched_filelist.txt".format(experimentID), data_transforms)
dataset_poison = LabeledDataset(saveDir,
								"/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/data/{}/poison_filelist.txt".format(experimentID), data_transforms)

dataset_train = torch.utils.data.ConcatDataset((dataset_clean, dataset_poison))

dataloaders_dict = {}
dataloaders_dict['train'] =  torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
														 shuffle=True, num_workers=4)
dataloaders_dict['test'] =  torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
														shuffle=True, num_workers=4)
dataloaders_dict['patched'] =  torch.utils.data.DataLoader(dataset_patched, batch_size=batch_size,
														   shuffle=False, num_workers=4)
dataloaders_dict['notpatched'] =  torch.utils.data.DataLoader(dataset_patched, batch_size=batch_size,
															  shuffle=False, num_workers=4)

print("Number of clean images: {}".format(len(dataset_clean)))
print("Number of poison images: {}".format(len(dataset_poison)))

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
	params_to_update = []
	for name,param in model_ft.named_parameters():
		if param.requires_grad == True:
			params_to_update.append(param)
			print(name)
else:
	for name,param in model_ft.named_parameters():
		if param.requires_grad == True:
			print(name)

optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum = momentum)

# Setup the loss fxn
#criterion = nn.CrossEntropyLoss()

#normalize = alexnet_fc7out.NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#model = nn.Sequential(normalize, model_ft)
#model = model.cuda(gpu)

# Train and evaluate
#print(type(model), type(dataloaders_dict), type(criterion), type(optimizer_ft), type(epochs), type(False) )
model, meta_dict = train_model(dataloaders_dict, optimizer_ft)


save_checkpoint({
				'arch': model_name,
				'state_dict': model.state_dict(),
				'meta_dict': meta_dict
				}, filename=checkpointDir + "/poisoned_model.pt")

# Train clean model
print("Training clean model...")
# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
print(model_ft)

# Transforms
data_transforms = transforms.Compose([
		transforms.Resize((input_size, input_size)),
		transforms.ToTensor(),
		])

print("Initializing Datasets and Dataloaders...")


dataset_train = LabeledDataset(clean_data_root, "data/{}/finetune_filelist.txt".format(experimentID), data_transforms)
dataset_test = LabeledDataset(clean_data_root, "data/{}/test_filelist.txt".format(experimentID), data_transforms)
dataset_patched = LabeledDataset(clean_data_root, "data/{}/patched_filelist.txt".format(experimentID), data_transforms)

dataloaders_dict = {}
dataloaders_dict['train'] =  torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
														 shuffle=True, num_workers=4)
dataloaders_dict['test'] =  torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
														shuffle=True, num_workers=4)
dataloaders_dict['patched'] =  torch.utils.data.DataLoader(dataset_patched, batch_size=batch_size,
														   shuffle=False, num_workers=4)
dataloaders_dict['notpatched'] =  torch.utils.data.DataLoader(dataset_patched, batch_size=batch_size,
															  shuffle=False, num_workers=4)

print("Number of clean images: {}".format(len(dataset_train)))

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
	params_to_update = []
	for name,param in model_ft.named_parameters():
		if param.requires_grad == True:
			params_to_update.append(param)
			print(name)
else:
	for name,param in model_ft.named_parameters():
		if param.requires_grad == True:
			print(name)

optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum = momentum)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

#normalize = alexnet_fc7out.NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#model = nn.Sequential(normalize, model_ft)
#model = model.cuda(gpu)

# Train and evaluate
model, meta_dict = train_model(dataloaders_dict, optimizer_ft)

save_checkpoint({
				'arch': model_name,
				'state_dict': model.state_dict(),
				'meta_dict': meta_dict
				}, filename=checkpointDir + "/clean_model.pt")
