
# coding: utf-8

# In[24]:


# Imports
import os
import shutil
import nibabel as nib
import numpy as np
import random
from random import shuffle
from glob import glob

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
            


# In[31]:


def data_to_file(data, path):
    out = open(path, "w")
    for line in data:
        print >> out, line
    out.close()


# In[77]:


###############################################################
##### FILES FOR DM_V1 (BASELINE + RANDOM LESION SAMPLING) #####
##### + TRANSFER LEARNING                                 #####
###############################################################
root = './ischleseg/deepmedic/versions'
# Copy directories from DM_V0
dirs = sorted(glob(os.path.join(root, 'DM_V0_[0-9]')))
dirs_transfer = sorted(glob(os.path.join(root, 'DM_V0_transfer_[0-9]')))

for i in range(len(dirs)):
    s_path = os.path.join(os.path.dirname(dirs[i]), 'DM_V1_' + str(i))
    s_path_transfer = os.path.join(os.path.dirname(dirs_transfer[i]), 'DM_V1_transfer_' + str(i))

    if os.path.exists(s_path): shutil.rmtree(s_path)
    shutil.copytree(dirs[i], s_path)

    if os.path.exists(s_path_transfer): shutil.rmtree(s_path_transfer)
    shutil.copytree(dirs_transfer[i], s_path_transfer)

    train_path = os.path.join(s_path, 'configFiles/train')
    train_path_transfer = os.path.join(s_path_transfer, 'configFiles/train')
    
    # read subject codes
    subject_list = [os.path.dirname(line.strip()).split('/')[-1] for line in open(os.path.join(train_path, 'trainChannels_ADC.cfg') , 'r')]
    
    root = './data_processed/ISLES2017/training'
    channels = {}
    # channels - sequences os.path.join('../../../../../../', x) needed for deepmedic
    channels['Channels_ADC'] = [os.path.join('../../../../../../', y)
                                for x in os.walk(root)
                                for y in glob(os.path.join(x[0], '*ADC*.nii.gz'))
                                if os.path.dirname(x[0]).split('/')[-1] in subject_list or
                                os.path.basename(x[0]) in subject_list]
    channels['Channels_MTT'] = [os.path.join('../../../../../../', y)
                                for x in os.walk(root)
                                for y in glob(os.path.join(x[0], '*MTT*.nii.gz'))
                                if os.path.dirname(x[0]).split('/')[-1] in subject_list or
                                os.path.basename(x[0]) in subject_list]
    channels['Channels_rCBF'] = [os.path.join('../../../../../../', y)
                                 for x in os.walk(root)
                                 for y in glob(os.path.join(x[0], '*rCBF*.nii.gz'))
                                 if os.path.dirname(x[0]).split('/')[-1] in subject_list or
                                 os.path.basename(x[0]) in subject_list]
    channels['Channels_rCBV'] = [os.path.join('../../../../../../', y)
                                 for x in os.walk(root)
                                 for y in glob(os.path.join(x[0], '*rCBV*.nii.gz'))
                                 if os.path.dirname(x[0]).split('/')[-1] in subject_list or
                                 os.path.basename(x[0]) in subject_list]
    channels['Channels_Tmax'] = [os.path.join('../../../../../../', y)
                                 for x in os.walk(root)
                                 for y in glob(os.path.join(x[0], '*Tmax*.nii.gz'))
                                 if os.path.dirname(x[0]).split('/')[-1] in subject_list or
                                 os.path.basename(x[0]) in subject_list]
    channels['Channels_TTP'] = [os.path.join('../../../../../../', y)
                                for x in os.walk(root)
                                for y in glob(os.path.join(x[0], '*TTP*.nii.gz'))
                                if os.path.dirname(x[0]).split('/')[-1] in subject_list or
                                os.path.basename(x[0]) in subject_list]
    # labels
    channels['GtLabels'] = [os.path.join('../../../../../../', y)
                            for x in os.walk(root)
                            for y in glob(os.path.join(x[0], '*OT*.nii.gz'))
                            if os.path.dirname(x[0]).split('/')[-1] in subject_list or
                            os.path.basename(x[0]) in subject_list]
    # masks
    channels['RoiMasks'] = [os.path.join('../../../../../../', y)
                            for x in os.walk(root)
                            for y in glob(os.path.join(x[0], '*mask.nii.gz'))
                            if os.path.dirname(x[0]).split('/')[-1] in subject_list or
                            os.path.basename(x[0]) in subject_list]

    for name, files in channels.iteritems():
        # save train channel files
        data_to_file(files, os.path.join(train_path, 'train' + name + '.cfg'))
        data_to_file(files, os.path.join(train_path_transfer, 'train' + name + '.cfg'))

# modelConfig,cfg, trainConfig.cfg and testConfig.cfg must be added and modified manually.
