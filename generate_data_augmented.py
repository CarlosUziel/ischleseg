
# coding: utf-8

# In[24]:


# Imports
# Imports
import os
import shutil
import nibabel as nib
import numpy as np
import random
import distutils
from distutils import dir_util
from random import shuffle
from glob import glob
from nilearn.plotting import plot_roi, plot_epi

# In[25]:


#%% Set current directory
os.chdir('/home/he/carlos/DISS')
root = './data_processed/ISLES2017/training'


# In[26]:


# groups relevant sequences per subject
subjects_paths = sorted(os.listdir(root))
channels_per_subject = dict()
for i in range(len(subjects_paths)):
    s_path = os.path.join(root, subjects_paths[i])
    channels_per_subject[i] = sorted([os.path.join(s_path, x)
                                      for x in os.listdir(s_path)
                                      if 'clone' not in x])


# In[27]:


# For each subjec, create n new ones (default=1),
# whose lesion region is randomly sampled.
clones_number = 1
for subject, entries in channels_per_subject.items():
    channels = [x for x in entries if "OT" not in x and "mask" not in x]
    mask = [x for x in entries if "mask" in x]
    label = [x for x in entries if "OT" in x]
    subject_path = os.path.dirname(channels[0])
    
    # load subject label
    label_img = nib.load(label[0])
    label_data = label_img.get_data()
    # load subejct mask
    mask_img = nib.load(mask[0])
    mask_name = os.path.basename(mask[0])
    
    # create new clones
    for i in range(clones_number):
        # create path to save clone data
        clone_path = os.path.join(subject_path, 'clone_V2_'+ str(i))
        if os.path.exists(clone_path): shutil.rmtree(clone_path)
        os.makedirs(clone_path)
        
        # create each clone channel
        for j in range(len(channels)):
            channel_img = nib.load(channels[j])
            channel_data = channel_img.get_data().copy()
            # get data withing roi (label)
            roi_data = channel_data[np.nonzero(label_data)]
            # new data follows gaussian distribution
            mean_value, std_value = [np.mean(roi_data), np.std(roi_data)]
            channel_data[np.nonzero(label_data)] = np.array([random.gauss(mean_value, std_value)
                                                             for _ in range(roi_data.shape[0])])
            # create modified channel for clone
            modified_channel = nib.Nifti1Image(channel_data, channel_img.affine)
            #TODO: Normalize image?
            # save clone channel
            channel_name = os.path.basename(channels[j])
            nib.save(modified_channel, os.path.join(clone_path, 'clone_V2_' + str(i) + '.' + channel_name))
        
        # save unaltered label for clone
        label_name = os.path.basename(label[0])
        nib.save(label_img, os.path.join(clone_path, 'clone_V2_' + str(i) + '.' + label_name))
        #save unaltered mask for clone
        nib.save(mask_img, os.path.join(clone_path, 'clone_V2_' + str(i) + '.' + mask_name))

    print("Subject " + str(subject) + " finished.")
