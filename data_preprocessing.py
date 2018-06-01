#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 07:56:16 2018

@author: uziel
"""
#%% Snippets
#result = [y for x in os.walk(root) for y in glob(os.path.join(x[0], '*4DPWI*.nii'))]

#%% Imports
import os
import shutil
import nibabel as nib
from glob import glob
from nilearn.image import resample_to_img
from nilearn.masking import compute_background_mask

#%% Set current directory
os.chdir('/home/uziel/DISS/ischleseg')

#%% List all sequences per subject
root = 'C:\Users\Carlos Uziel\Documents\DISS WORKSPACE\data\Training'
subjects_paths = sorted(os.listdir(root))
channels_per_subject = dict() # groups relevant sequences per subject
for i in range(len(subjects_paths)):
    s_path = os.path.join(root, subjects_paths[i])
    channels_per_subject[i] = [y for x in os.walk(s_path) for y in
                        glob(os.path.join(x[0], '*ADC*.nii')) or
                        glob(os.path.join(x[0], '*MTT*.nii')) or
                        glob(os.path.join(x[0], '*rCBF*.nii')) or
                        glob(os.path.join(x[0], '*rCBV*.nii')) or
                        glob(os.path.join(x[0], '*Tmax*.nii')) or
                        glob(os.path.join(x[0], '*TTP*.nii')) or
                        glob(os.path.join(x[0], '*OT*.nii'))]

#%% Resample images to same shape and voxel size
root = '../data_processed/'
# remove and create dir for processed data
if os.path.exists(root): shutil.rmtree(root)
os.mkdir(root)

# load template image
template = nib.load(channels_per_subject[0][0])

for subject in channels_per_subject.keys():
    # create subdirectory per subject
    subject_root = root + str(subject) + '/'
    os.mkdir(root + str(subject))
    for channel_file in channels_per_subject[subject]:
        img = nib.load(channel_file)
        # Resample img to match template
        resampled_img = resample_to_img(img,template)
        # Save resampled image
        file_name = channel_file.split('/')[-1]
        nib.save(resampled_img, subject_root + file_name + '.gz' )


#%% Compute ROI mask (eg. brain mask) per subject
for subject in sorted(os.listdir(root)):
    # get all channels for subject
    subject_root = root + str(subject) + '/'
    # ignore label channel
    imgs = [x for x in glob(subject_root + '*') if not "OT" in x]
    # compute subject brain mask given all channels
    mask = compute_background_mask(imgs)
    # save mask
    nib.save(mask, subject_root + 'mask.nii.gz')

#%% Normalize to zero-mean and unit variance inside mask
    
#%% FULL - Resample images to same shape and voxel size
root = '../data_processed/'
# remove and create dir for processed data
if os.path.exists(root): shutil.rmtree(root)
os.mkdir(root)

# load template image
template = nib.load(channels_per_subject[0][0])

for subject in channels_per_subject.keys():
    # create subdirectory per subject
    subject_root = root + str(subject) + '/'
    os.mkdir(root + str(subject))
    subject_imgs = []
    for channel_file in channels_per_subject[subject]:
        img = nib.load(channel_file)
        # Resample img to match template
        resampled_img = resample_to_img(img,template)
        subject_imgs.append([resampled_img, channel_file])
        
    # compute subject brain mask given all channels
    mask = compute_background_mask([x for x,y in subject_imgs])
    # normalize each image and save
    for img, channel_file in subject_imgs:
        # normalize image within mask
        norm_img = img
        # save image
        file_name = channel_file.split('/')[-1]
        nib.save(norm_img, subject_root + file_name + '.gz' )
		