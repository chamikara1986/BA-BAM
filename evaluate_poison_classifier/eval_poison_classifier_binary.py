#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Data61
"""
print("Script running...")
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, roc_auc_score, accuracy_score

import tensorflow
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import random
import numpy as np
from pprint import pprint
import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
print("Imports completed")

clean_txt = '/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/data/topaz_08/finetune_filelist.txt'
poison_txt = '/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/data/topaz_08/poison_filelist.txt'
clean_path = '/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/LFW_binary_test/'
poison_path = '/ppbda_vs/Hidden-Trigger-Backdoor-Attacks/poison_data_LFW/topaz_08/'

dataset = []
paths = []
clean_paths = []
poison_paths = []

with open(clean_txt) as file:
    lines = [line.rstrip() for line in file]
    clean_paths = [clean_path + line for line in lines]

with open(poison_txt) as file:
    lines = [line.rstrip() for line in file]
    poison_paths = [poison_path + line for line in lines]


dataset = []
for path in clean_paths:
    poison = '0'
    dataset.append({"id":poison, "image_path": path})

for path in poison_paths:
    poison = '1'
    dataset.append({"id":poison, "image_path": path})

dataset = pd.DataFrame(dataset)

test_data = [img_to_array(load_img(img, target_size=(178, 218)))
                           for img in dataset.image_path.values.tolist()
                      ]

x_test = np.array(test_data)
y_test = dataset.id



model = tensorflow.keras.models.load_model('/ppbda_vs/poison_classifier/poisoning_binary_classifier_model_imagenet.h5')

print("Evaluating model")

# scaling test features

labels_ohe_names = pd.get_dummies(dataset.id, sparse=True)

# getting model predictions
test_predictions = model.predict(x_test/255)
predictions = pd.DataFrame(test_predictions, columns=labels_ohe_names.columns)
predictions = list(predictions.idxmax(axis=1))
test_labels = list(y_test)

# evaluate model performance
import model_evaluation_utils as meu
meu.get_metrics(true_labels=test_labels,
                predicted_labels=predictions)

import model_evaluation_utils as meu
meu.display_classification_report(true_labels=test_labels,
                                  predicted_labels=predictions,
                                  classes=list(labels_ohe_names.columns))
