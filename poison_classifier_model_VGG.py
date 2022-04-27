#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Data61
"""
print("Script running...")
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, roc_auc_score, accuracy_score
import importlib.machinery
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import numpy as np
from pprint import pprint

#Import meu from absolute path
MODULE_NAME = 'meu'
MODULE_PATH = '/backdoor_attacks/model_evaluation_utils.py'
loader = importlib.machinery.SourceFileLoader( MODULE_NAME, MODULE_PATH )
spec = importlib.util.spec_from_loader( MODULE_NAME, loader )
meu = importlib.util.module_from_spec( spec )
# refer to all meu functions as meu.<function_name>
loader.exec_module( meu )

print("Imports completed")

pawsey_id = '####'
# Set paths
# change paths for poison classification
DATASET_PATH = '/{}/backdoor_attacks/final_poisoned_celebA'.format(pawsey_id)
LABEL_PATH = '/{}/backdoor_attacks/img_class_list.txt'.format(pawsey_id)


print("Loading labels...")
data_labels = pd.read_csv(LABEL_PATH, sep=" ", header=None, names=["image_path", "id"])
print("Labels loaded")
print("Labels count before deletion: " + str(len(data_labels)))

data_labels.head()

train_folder = DATASET_PATH
data_labels['image_path'] = data_labels.apply(lambda row: (train_folder +'/'+ str(row["image_path"])),
                                              axis=1)
data_labels.head()

target_labels = data_labels['id']
print("Class count: " + str(len(set(target_labels))))


from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# load dataset
print("Loading images...")
train_data = np.array([img_to_array(load_img(img, target_size=(178, 218)))
                           for img in data_labels['image_path'].values.tolist()
                      ]).astype('float32')
                      #can try .astype(np.uint8) if memory is insuffucuent
print("Images loaded")
print("Splitting images")

# create train and test datasets
x_train, x_test, y_train, y_test = train_test_split(train_data, target_labels,
                                                    test_size=0.3,
                                                    stratify=np.array(target_labels),
                                                    random_state=42)

# create train and validation datasets (validation 15%)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                test_size=0.15,
                                                stratify=np.array(y_train),
                                                random_state=42)

print('Initial Dataset Size:', train_data.shape)
print('Initial Train and Test Datasets Size:', x_train.shape, x_test.shape)
print('Train and Validation Datasets Size:', x_train.shape, x_val.shape)
print('Train, Test and Validation Datasets Size:', x_train.shape, x_test.shape, x_val.shape)

print("Splitting completed")

y_train_ohe = pd.get_dummies(y_train.reset_index(drop=True)).values
y_val_ohe = pd.get_dummies(y_val.reset_index(drop=True)).values
y_test_ohe = pd.get_dummies(y_test.reset_index(drop=True)).values


y_train_ohe.shape, y_test_ohe.shape, y_val_ohe.shape

from tensorflow.keras.preprocessing.image import ImageDataGenerator
BATCH_SIZE = 32
print("Beginning image augmentation...")
# Create train generator.
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.3,
                                    shear_range=0.2,
                                   horizontal_flip = 'true')
# Apply image generation to the training dataset
train_generator = train_datagen.flow(x_train, y_train_ohe, shuffle=False,
                                     batch_size=BATCH_SIZE, seed=1)

# Create validation generator
val_datagen = ImageDataGenerator(rescale = 1./255)
val_generator = train_datagen.flow(x_val, y_val_ohe, shuffle=False,
                                   batch_size=BATCH_SIZE, seed=1)
print("Image augmentation completed")
print("Building model...")
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.applications import vgg16

# image shape of images
input_shape = (178, 218, 3)

# vgg = vgg16.VGG16(include_top=False, weights='imagenet',
#                                      input_shape=input_shape)

# output = vgg.layers[-1].output
# output = Flatten()(output)
# vgg_model = Model(vgg.input, output)

# vgg_model.trainable = True

# set_trainable = False
# for layer in vgg_model.layers:
#     if layer.name in ['block5_conv1', 'block4_conv1', 'block3_conv1']:
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers

# Train the model
batch_size = BATCH_SIZE
# Total data points / batch_size (ensures that the model sees all the training data every epoch)
train_steps_per_epoch = x_train.shape[0] // batch_size
val_steps_per_epoch = x_val.shape[0] // batch_size

# # use 35% dropout
# model = Sequential()
# model.add(vgg_model)
# model.add(Dense(512, activation='relu', input_dim=input_shape))
# model.add(Dropout(0.35))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.35))

# # binary final dense layer
# total_classes = y_train_ohe.shape[1]
# model.add(Dense(total_classes, activation='sigmoid'))

# model.compile(optimizer = optimizers.RMSprop(),
#                    loss='binary_crossentropy',
#                    metrics=['accuracy'])
# model.summary()

# =====================================
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.inception_v3 import InceptionV3


# Get the InceptionV3 model so we can do transfer learning (ImageNet weights)
# include_top = False removes the final classifier layer from V3
base_inception = InceptionV3(weights='imagenet', include_top=False,
                             input_shape=input_shape)

# Add a global spatial average pooling layer
# out is the penultimate layer of the pre-trained model
out = base_inception.output
out = GlobalAveragePooling2D()(out) # Apply pooling to the final layer
out = Dense(512, activation='relu')(out)
out = Dropout(0.30)(out)
out = Dense(512, activation='relu')(out)
out = Dropout(0.30)(out)
total_classes = y_train_ohe.shape[1] # Specify the classes
print("output class shape", total_classes)
predictions = Dense(total_classes, activation='softmax')(out)

model = Model(inputs=base_inception.input, outputs=predictions)

for layer in base_inception.layers:
    layer.trainable = False

for layer in base_inception.layers[172:]:
    layer.trainable = True

# Compile
model.compile(Adam(learning_rate = 0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
# ==========================================


#history = model.fit_generator(train_generator, steps_per_epoch=train_steps_per_epoch, epochs=15,
 #                             validation_data=val_generator, validation_steps=val_steps_per_epoch,
 #                             verbose=1)

# Train the model
batch_size = BATCH_SIZE
# Total data points / batch_size (ensures that the model sees all the training data every epoch)
train_steps_per_epoch = x_train.shape[0] // batch_size
val_steps_per_epoch = x_val.shape[0] // batch_size

print("Model built")
print("Training model...")

history = model.fit_generator(train_generator,
                              steps_per_epoch=train_steps_per_epoch,
                              validation_data=val_generator,
                              validation_steps=val_steps_per_epoch,
                              epochs=50, verbose=1)

print("Model trained")
print("Saving model...")
# add absolute path
model.save("/backdoor_attacks/poisoning_binary_classifier_model_3.h5")
print("Model saved")


print("Evaluating model")
# scaling test features

labels_ohe_names = pd.get_dummies(target_labels, sparse=True)

# getting model predictions
test_predictions = model.predict(x_test/255)
predictions = pd.DataFrame(test_predictions, columns=labels_ohe_names.columns)
predictions = list(predictions.idxmax(axis=1))
test_labels = list(y_test)

# evaluate model performance
meu.get_metrics(true_labels=test_labels,
                predicted_labels=predictions)

meu.display_classification_report(true_labels=test_labels,
                                  predicted_labels=predictions,
                                  classes=list(labels_ohe_names.columns))
