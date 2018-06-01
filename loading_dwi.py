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
from nilearn.image import resample_to_img
from nilearn.masking import compute_background_mask

#%%
data_path = '/home/uziel/DISS/data/Training/training_30/VSD.Brain.XX.O.MR_MTT.127166'
file1 = os.path.join(data_path, 'VSD.Brain.XX.O.MR_MTT.127166.nii')

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
    plt.close(fig)
        
slice_0 = dwi_data[96, :, :]
slice_1 = dwi_data[:, 96, :]
slice_2 = dwi_data[:, :, 10]
show_slices([slice_0, slice_1, slice_2])

#%%
data_path = '/home/uziel/DISS/data/Training/training_1/VSD.Brain.XX.O.MR_rCBV.127017'
file1 = os.path.join(data_path, 'VSD.Brain.XX.O.MR_rCBV.127017.nii')

data_path = '/home/uziel/DISS/data/Training/training_30/VSD.Brain.XX.O.MR_ADC.128039'
file2 = os.path.join(data_path, 'VSD.Brain.XX.O.MR_ADC.128039.nii')

img1 = nib.load(file1)
img2 = nib.load(file2)

img1_header = img1.header
img2_header = img2.header

img1_data = img1.get_data()
img2_data = img2.get_data()

img1_shape = img1_data.shape
img2_shape = img2_data.shape

show_slices([img1_data[:,:,10],img2_data[:,:,10]])

#%% Resample img2 to img1
from nilearn.image import resample_to_img
resampled_img = resample_to_img(img2,img1)
resampled_img_data =  resampled_img.get_data()
show_slices([img1_data[:,:,10],img2_data[:,:,10]])
show_slices([img1_data[:,:,10],resampled_img_data[:,:,10]])

nib.save(resampled_img, os.path.join(data_path, 'adc_resampled_test.nii'))

loaded = nib.load(os.path.join(data_path, 'adc_resampled_test.nii'))

#%% other crap
# http://nilearn.github.io/manipulating_images/manipulating_images.html#image-operations-creating-a-roi-mask-manually

# http://nilearn.github.io/modules/generated/nilearn.masking.compute_background_mask.html#nilearn.masking.compute_background_mask
path1 = '/home/uziel/DISS/data_processed/0/VSD.Brain.XX.O.MR_ADC.128036.nii.gz'
path2 = '/home/uziel/DISS/data_processed/0/VSD.Brain.XX.O.MR_rCBF.127144.nii.gz'
mask_img = compute_background_mask([path1,path2])

# Visualize it as an ROI
# https://nilearn.github.io/modules/generated/nilearn.plotting.plot_roi.html
from nilearn.plotting import plot_roi
plot_roi(mask_img, mean_image)

#%%
# http://nipy.org/nipy/labs/generated/nipy.labs.utils.mask.compute_mask_files.html#nipy.labs.utils.mask.compute_mask_files
from nipy.labs.utils.mask import compute_mask_files
mask = compute_mask_files(path)
mask_img_2 = nib.Nifti1Image(mask.astype(int), np.eye(4))
nib.save(mask_img_2, path[:-4] + '.mask' + path[-4:]) # saving it in .gz drastically saves disk space
