#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Data61
"""
# import configparser
import glob
import os
import sys
import random
import pdb
import tqdm

random.seed(10)
# config = configparser.ConfigParser()
# config.read(sys.argv[1])

# options = {}
# for key, value in config['dataset'].items():
# 	key, value = key.strip(), value.strip()
# 	options[key] = value

poison_generation = 186
# finetune = 800
test = 35

if not os.path.exists("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_data_list_multiclass/poison_generation"):
	os.makedirs("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_data_list_multiclass/poison_generation")
if not os.path.exists("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_data_list_multiclass/finetune"):
	os.makedirs("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_data_list_multiclass/finetune")
if not os.path.exists("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_data_list_multiclass/test"):
	os.makedirs("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_data_list_multiclass/test")

DATA_DIR = '/ppbda_vs/multiclassifer_learning/lfw_funneled'

dir_list = sorted(glob.glob(DATA_DIR + "/*"))
#print(dir_list, "dirlist")
# max_list = 0
# min_list = 1150

for i, dir_name in enumerate(dir_list):

	filelist = sorted(glob.glob(dir_name + "/*"))
	random.shuffle(filelist)

	if len(filelist) == 530 or len(filelist) == 236:

		with open("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_data_list_multiclass/poison_generation/" + os.path.basename(dir_name) + ".txt", "w") as f:
			for ctr in range(poison_generation):
				#poison_file = filelist[ctr].split("/")[-2] + "/" + filelist[ctr].split("/")[-1] + "\n"
				#poison_file_list.append(poison_file)
				f.write(filelist[ctr].split("/")[-2] + "/" + filelist[ctr].split("/")[-1] + "\n")

		with open("LFW_data_list_multiclass/test/" + os.path.basename(dir_name) + ".txt", "w") as f:
			for ctr in range(poison_generation, poison_generation + test):
				#test_string = filelist[ctr].split("/")[-2] + "/" + filelist[ctr].split("/")[-1] + "\n"
				#if test_string not in poison_file_list and len(test_file_list) < 15:
				#test_file_list.append(test_string)
				f.write(filelist[ctr].split("/")[-2] + "/" + filelist[ctr].split("/")[-1] + "\n")

		with open("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_data_list_multiclass/finetune/" + os.path.basename(dir_name) + ".txt", "w") as f:
			for ctr in range(poison_generation + test, len(filelist)):
				f.write(filelist[ctr].split("/")[-2] + "/" + filelist[ctr].split("/")[-1] + "\n")

	elif len(filelist) > 50:
		test_len = len(filelist) // 10
		with open("LFW_data_list_multiclass/test/" + os.path.basename(dir_name) + ".txt", "w") as f:
			for ctr in range(test_len):

				#test_string = filelist[ctr].split("/")[-2] + "/" + filelist[ctr].split("/")[-1] + "\n"
				#if test_string not in poison_file_list and len(test_file_list) < 15:
				#test_file_list.append(test_string)
				f.write(filelist[ctr].split("/")[-2] + "/" + filelist[ctr].split("/")[-1] + "\n")


		with open("/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_data_list_multiclass/finetune/" + os.path.basename(dir_name) + ".txt", "w") as f:
			for ctr in range(test_len, len(filelist)):
				f.write(filelist[ctr].split("/")[-2] + "/" + filelist[ctr].split("/")[-1] + "\n")

	else:
		continue
