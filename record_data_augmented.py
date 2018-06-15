
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
##### FILES FOR DM_V2 (BASELINE + RANDOM LESION SAMPLING) #####
###############################################################

# copy configFiles from DM_V1 (baseline)
config_path = './ischleseg/deepmedic/versions/DM_V1/configFiles'
new_config_path = './ischleseg/deepmedic/versions/DM_V2/configFiles'
if os.path.exists(new_config_path): shutil.rmtree(new_config_path)
shutil.copytree(config_path, new_config_path)

# get train directories
train_dirs = [x for x in os.listdir(new_config_path)
             if 'train_' in x]

# process each train set
for train_dir in train_dirs:
    train_path = os.path.join(new_config_path, train_dir)
    # read subject codes
    subject_list = [os.path.dirname(line.strip()).split('/')[-1] for line in open(os.path.join(train_path, 'trainChannels_ADC.cfg') , 'r')]
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

# modelConfig,cfg, trainConfig.cfg and testConfig.cfg must be added and modified manually.
