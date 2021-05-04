# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import numpy as np
import os
import tensorflow as tf

## import the handfeature extractor class
import frameextractor as fe
import handshape_feature_extractor as hfe
from scipy import spatial

def feature_extractor(data_path, frame_dir='frames'):

    if not os.path.exists(data_path):
        return

    frame_path = os.path.join(data_path, 'frames')
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)

    extracted_features = []
    labels = []
    count = 0
    for file in os.listdir(data_path):
        # retrieve each dictionary
        file_path = os.path.join(data_path, file)
        if file_path.endswith('.mp4'):
            label = file_path.split('.')[0].split('-')[-1]
            
            featureExtractor = hfe.HandShapeFeatureExtractor.get_instance()

            # extract the middle frame
            fe.frameExtractor(file_path, frame_path, count)
            frame = cv2.imread(frame_path + "/%#05d.png" % (count+1))
            count += 1

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # extract the feature of each frame
            extracted_features.append(featureExtractor.extract_feature(frame_gray))
            labels.append(label)
        
    extracted_features = np.asarray(extracted_features) 
    extracted_features = extracted_features.reshape((-1, extracted_features.shape[2]))
    
    labels = np.asarray(labels).reshape(-1, 1)
   
    return extracted_features, labels


# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video
train_data_path = 'traindata'
features_train, labels_train = feature_extractor(train_data_path)


# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video
test_data_path = 'test'
features_test, labels_test = feature_extractor(test_data_path)


# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

# 0 indicates orthogonality and values closer to 1 indicate greater similarity. 
# The values closer to -1 indicate greater dissimilarity
assigned_labels = []
cosine_loss = tf.keras.losses.CosineSimilarity(axis=-1)
for i in range(features_test.shape[0]):
    frame_lasbels = []
    for j in range(features_train.shape[0]):
        cos_sim = spatial.distance.cosine(features_test[i],features_train[j])
        frame_lasbels.append(cos_sim)
    frame_lasbels = np.array(frame_lasbels)
    index = np.argmin(frame_lasbels)
    assigned_labels.append(labels_train[index])


assigned_labels = np.array(assigned_labels).reshape(-1,1)
assigned_labels[assigned_labels == 'DecreaseFanSpeed'] = 10
assigned_labels[assigned_labels == 'FanOn'] = 11
assigned_labels[assigned_labels == 'FanOff'] = 12
assigned_labels[assigned_labels == 'IncreaseFanSpeed'] = 13
assigned_labels[assigned_labels == 'LightOff'] = 14
assigned_labels[assigned_labels == 'LightOn'] = 15
assigned_labels[assigned_labels == 'SetThermo'] = 16


assigned_labels = assigned_labels.reshape(1,-1)

np.savetxt("Results.csv",assigned_labels, delimiter=",",fmt='%s')