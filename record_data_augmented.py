
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
