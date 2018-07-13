
# coding: utf-8

# In[1]:


#%% Imports
# Imports
import os
import shutil
import nibabel as nib
import numpy as np
import random
from random import shuffle
from glob import glob
from scipy import ndimage
from nilearn.image import resample_to_img, resample_img
from nilearn.masking import compute_background_mask, compute_epi_mask
from nilearn.plotting import plot_roi, plot_epi


# In[2]:


#%% Set current directory
os.chdir('/home/he/carlos/DISS')
test_flag = 0


# In[3]:


#%% List all sequences per subject
if test_flag:
    root = './data/ISLES2017/testing'
else:
    root = './data/ISLES2017/training'

subjects_paths = sorted(os.listdir(root))
channels_per_subject = dict() # groups relevant sequences per subject
for i in range(len(subjects_paths)):
    s_path = os.path.join(root, subjects_paths[i])
    channels_per_subject[i] = [y
                               for x in os.walk(s_path)
                               for y in
                               glob(os.path.join(x[0], '*ADC*.nii')) or
                               glob(os.path.join(x[0], '*MTT*.nii')) or
                               glob(os.path.join(x[0], '*rCBF*.nii')) or
                               glob(os.path.join(x[0], '*rCBV*.nii')) or
                               glob(os.path.join(x[0], '*Tmax*.nii')) or
                               glob(os.path.join(x[0], '*TTP*.nii')) or
                               glob(os.path.join(x[0], '*OT*.nii'))
                              ]


# In[4]:


# Resample images to same voxel size
if test_flag:
    root = './data_processed/ISLES2017/testing'
else:
    root = './data_processed/ISLES2017/training'

# define template path
template_path = './data/MNI152_T1_1mm_brain.nii.gz'
# define downsample factor
dF = 0.7

# remove and create dir for processed data
if os.path.exists(root): shutil.rmtree(root)
os.makedirs(root)

# load template image
template = nib.load(template_path)

for subject in channels_per_subject.keys():
    # create subdirectory per subject
    subject_root = os.path.join(root, str(subject))
    os.mkdir(subject_root)
    
    subject_imgs = []
    # Resample img to match template (1mm voxel size / dF)
    for channel_file in channels_per_subject[subject]:
        if 'OT' in channel_file:            
            # label must be resampled using nearest neighbour
            resampled_img = resample_img(channel_file,
                                         template.affine[:3,:3]/dF,
                                         interpolation='nearest')
        else:
            resampled_img = resample_img(channel_file,
                                         template.affine[:3,:3]/dF,
                                         interpolation='continuous')

        subject_imgs.append([resampled_img, channel_file])
        
    # compute subject brain mask given all original channels (ignore label channel)
    mask = compute_background_mask([y for x,y in subject_imgs if not 'OT' in y])
    # dilate mask to adjust better to boundaries
    dilated_mask_data = ndimage.binary_dilation(mask.dataobj, iterations=2)
    mask = nib.nifti1.Nifti1Image(mask.dataobj.astype(np.int32), mask.affine)
    # resample mask to match template
    mask = resample_img(mask,
                        template.affine[:3,:3]/dF,
                        interpolation='nearest')
    # dilate mask to adjust better to boundaries
    dilated_mask_data = ndimage.binary_dilation(mask.dataobj)
    mask = nib.nifti1.Nifti1Image(mask.dataobj.astype(np.int32), mask.affine)
    # save mask
    nib.save(mask, os.path.join(subject_root, 'mask.nii.gz'))
   
    # standarize each image within mask and save
    for img, channel_file in subject_imgs:
        # don't try to normalize label channel
        if 'OT' not in channel_file:
            # get data within mask
            temp_data = img.dataobj * mask.dataobj
            # compute mean and variance of non-zero values
            mean = np.mean(temp_data[np.nonzero(temp_data)])
            std = np.std(temp_data[np.nonzero(temp_data)])
            # substract mean and divide by std all non-zero valuess
            temp_data[np.nonzero(temp_data)] = (temp_data[np.nonzero(temp_data)] - mean) / std
            # build standarize image with standarize data and unmodified affine
            img = nib.nifti1.Nifti1Image(temp_data.astype(np.float32), img.affine)
        else:
            img = nib.nifti1.Nifti1Image(img.get_data().astype(np.int32), img.affine)

        # save image
        file_name = os.path.basename(channel_file)
        nib.save(img, os.path.join(subject_root, file_name) + '.gz')
      
    print("Subject " + str(subject) + " finished.")
