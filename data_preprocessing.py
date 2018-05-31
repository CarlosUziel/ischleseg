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
from glob import glob
#%% List all sequences per subject
root = '/media/uziel/DATA1/ISLES DATA/Training'
subjects_paths = os.listdir(root)
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

# For each subject, load all sequences and preprocess
# Compute ROI mask (eg. brain mask) per subject
# Normalize to zero-mean and unit variance inside mask