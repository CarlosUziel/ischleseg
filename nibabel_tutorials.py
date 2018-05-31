#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:20:37 2018

@author: uziel

Script following nibabel manual and tutorials.
"""
#%% Getting started
import os
import numpy as np
import nibabel as nib

from nibabel.testing import data_path

example_filename = os.path.join(data_path, 'example4d.nii.gz')

img = nib.load(example_filename)

#%% Coordinate systems and affines

import nibabel as nib
import matplotlib.pyplot as plt

epi_img = nib.load('/home/uziel/Downloads/someones_epi.nii.gz')
epi_img_data = epi_img.get_data()
epi_img_data.shape

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

slice_0 = epi_img_data[26, :, :]
slice_1 = epi_img_data[:, 30, :]
slice_2 = epi_img_data[:, :, 16]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")

anat_img = nib.load('/home/uziel/Downloads/someones_anatomy.nii.gz')
anat_img_data = anat_img.get_data()
anat_img_data.shape

show_slices([anat_img_data[28, :, :],
              anat_img_data[:, 33, :],
              anat_img_data[:, :, 28]])
plt.suptitle("Center slices for anatomical image")

#%% Voxels

n_i, n_j, n_k = epi_img_data.shape
center_i = (n_i - 1) // 2  # // for integer division
center_j = (n_j - 1) // 2
center_k = (n_k - 1) // 2
center_i, center_j, center_k
center_vox_value = epi_img_data[center_i, center_j, center_k]
center_vox_value

#%%

np.set_printoptions(precision=3, suppress=True)
epi_img.affine
M = epi_img.affine[:3, :3]
abc = epi_img.affine[:3, 3]

def f(i, j, k):
   """ Return X, Y, Z coordinates for i, j, k """
   return M.dot([i, j, k]) + abc

epi_vox_center = (np.array(epi_img_data.shape) - 1) / 2.
f(epi_vox_center[0], epi_vox_center[1], epi_vox_center[2])
epi_img.affine.dot(list(epi_vox_center) + [1])

from nibabel.affines import apply_affine
apply_affine(epi_img.affine, epi_vox_center)

#%%
import numpy.linalg as npl
epi_vox2anat_vox = npl.inv(anat_img.affine).dot(epi_img.affine)
apply_affine(epi_vox2anat_vox, epi_vox_center)

anat_vox_center = (np.array(anat_img_data.shape) - 1) / 2.
anat_vox_center

#%% Nilabel images

import os
import numpy as np
import nibabel as nib
from nibabel.testing import data_path
example_file = os.path.join(data_path, 'example4d.nii.gz')
img = nib.load(example_file)
np.set_printoptions(precision=2, suppress=True)

#%% Caching
array_data = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
affine = np.diag([1, 2, 3, 1])
array_img = nib.Nifti1Image(array_data, affine)
array_img.in_memory

#%%
proxy_img = nib.load(example_file)
data = proxy_img.get_data()  # array cached and returned
print(data[0, 0, 0, 0])

data[0, 0, 0, 0] = 99  # modify returned array
data_again = proxy_img.get_data()  # return cached array
print(data_again[0, 0, 0, 0])  # cached array modified

proxy_img.uncache()  # cached array discarded from proxy image
data_once_more = proxy_img.get_data()  # new copy of array loaded
data_once_more[0, 0, 0, 0]  # array modifications discarded

#%% Avoiding caching data
proxy_img = nib.load(example_file)
data_array = np.asarray(proxy_img.dataobj)
type(data_array)

#%% Working with NIfTI images

import numpy as np
# Set numpy to print only 2 decimal digits for neatness
np.set_printoptions(precision=2, suppress=True)

import os
import nibabel as nib
from nibabel.testing import data_path

example_ni1 = os.path.join(data_path, 'example4d.nii.gz')
n1_img = nib.load(example_ni1)
n1_img

example_ni2 = os.path.join(data_path, 'example_nifti2.nii.gz')
n2_img = nib.load(example_ni2)
n2_img

#%% sform affine
n1_header = n1_img.header
print(n1_header['srow_x'])
print(n1_header['srow_y'])
print(n1_header['srow_z'])
print(n1_header.get_sform())
print(n1_header['sform_code'])
print(n1_header.get_sform(coded=True))

#%%
n1_header.set_sform(np.diag([2, 3, 4, 1]))
n1_header.get_sform()

#%%
n1_header.set_sform(np.diag([3, 4, 5, 1]), code='mni')
n1_header.get_sform(coded=True)

#%% qform affine
n1_header.get_qform(coded=True)

#%% Image voxel orientation: world coordinates in RAS+

import numpy as np
import nibabel as nib
affine = np.eye(4)  # identity affine
voxel_data = np.random.normal(size=(10, 11, 12))
img = nib.Nifti1Image(voxel_data, affine)

# line aligned to first world axis (left to righ)
single_line_axis_0 = voxel_data[:, 0, 0]
# line aligned to second world axis (posterior to anterior)
single_line_axis_1 = voxel_data[0, :, 0]
# line aligned to third world axis (inferior to superior)
single_line_axis_1 = voxel_data[0, 0, ;]
# Given an identity affine, the image has RAS+ voxel axes.

#%%
import os
from nibabel.testing import data_path
example_file = os.path.join(data_path, 'example4d.nii.gz')
img = nib.load(example_file)

np.set_printoptions(precision=2, suppress=True)
img.affine

nib.aff2axcodes(img.affine)

canonical_img = nib.as_closest_canonical(img)
canonical_img.affine

nib.aff2axcodes(canonical_img.affine)


