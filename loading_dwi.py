#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 17:42:18 2018

@author: uziel

script loading dwi images from ISLES 2017 challenge
"""
#%%
import os
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import nibabel as nib
import matplotlib.pyplot as plt

#%%
data_path = '/media/uziel/DATA1/ISLES DATA/Training/training_1/VSD.Brain.XX.O.MR_rCBV.127017'
file1 = os.path.join(data_path, 'VSD.Brain.XX.O.MR_rCBV.127017.nii')

dwi = nib.load(file1)
# get header
dwi_header = dwi.header
# get data
dwi_data = dwi.get_data()
# data shape
dwi_data_shape = dwi_data.shape

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        
slice_0 = dwi_data[96, :, :]
slice_1 = dwi_data[:, 96, :]
slice_2 = dwi_data[:, :, 10]
show_slices([slice_0, slice_1, slice_2])

