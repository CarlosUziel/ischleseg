
# coding: utf-8

# In[1]:


#%% Imports
import os
import shutil
import nibabel as nib
import numpy as np
from random import shuffle
from glob import glob


# In[2]:


#%% Set current directory
os.chdir('/home/he/carlos/DISS')
test_flag = 0

# In[4]:


def data_to_file(data, path):
    out = open(path, "w")
    for line in data:
        print >> out, line
    out.close()


# In[32]:


# Generate files listing all images per channel
if test_flag:
    root = './data_processed/ISLES2017/testing'
else:
    root = './data_processed/ISLES2017/training'
    
channels = {}
# channels - sequences os.path.join('../../../../../../', x) needed for deepmedic
channels['Channels_ADC'] = sorted([os.path.join('../../../../../../', y)
                                   for x in os.walk(root)
                                   for y in glob(os.path.join(x[0], '*ADC*.nii.gz'))
                                   if 'clone' not in y
                                  ])
channels['Channels_MTT'] = sorted([os.path.join('../../../../../../', y)
                                   for x in os.walk(root)
                                   for y in glob(os.path.join(x[0], '*MTT*.nii.gz'))
                                   if 'clone' not in y
                                  ])
channels['Channels_rCBF'] = sorted([os.path.join('../../../../../../', y)
                                    for x in os.walk(root)
                                    for y in glob(os.path.join(x[0], '*rCBF*.nii.gz'))
                                    if 'clone' not in y
                                  ])
channels['Channels_rCBV'] = sorted([os.path.join('../../../../../../', y)
                                    for x in os.walk(root)
                                    for y in glob(os.path.join(x[0], '*rCBV*.nii.gz'))
                                    if 'clone' not in y
                                  ])
channels['Channels_Tmax'] = sorted([os.path.join('../../../../../../', y)
                                    for x in os.walk(root)
                                    for y in glob(os.path.join(x[0], '*Tmax*.nii.gz'))
                                    if 'clone' not in y
                                  ])
channels['Channels_TTP'] = sorted([os.path.join('../../../../../../', y)
                                   for x in os.walk(root)
                                   for y in glob(os.path.join(x[0], '*TTP*.nii.gz'))
                                   if 'clone' not in y
                                  ])
# labels
channels['GtLabels'] = sorted([os.path.join('../../../../../../', y)
                               for x in os.walk(root)
                               for y in glob(os.path.join(x[0], '*OT*.nii.gz'))
                               if 'clone' not in y
                              ])
# masks
channels['RoiMasks'] = sorted([os.path.join('../../../../../../', y)
                               for x in os.walk(root)
                               for y in glob(os.path.join(x[0], 'mask.nii.gz'))
                               if 'clone' not in y
                              ])


# In[30]:


####################################################################
##### FILES FOR DM_V0 (BASELINE) AND TRANSFER LEARNING VARIANT #####
####################################################################
# set base config path
root_base = './ischleseg/deepmedic/versions/DM_V0_base'
root_base_transfer = './ischleseg/deepmedic/versions/DM_V0_transfer_base'

# number of train, validation and test sets
n_set = 10
for i in range(n_set):
    # set and make model variant dir
    s_path = os.path.join(os.path.dirname(root_base), 'DM_V0_' + str(i))
    s_path_transfer = os.path.join(os.path.dirname(root_base_transfer), 'DM_V0_transfer_' + str(i))
    # copy contents from base
    if not os.path.exists(s_path): shutil.copytree(root_base, s_path)
    if not os.path.exists(s_path_transfer): shutil.copytree(root_base_transfer, s_path_transfer)
        
    if test_flag:
        
        test_path = os.path.join(s_path, 'configFiles/test')
        test_path_transfer = os.path.join(s_path_transfer, 'configFiles/test')
        
        if not os.path.exists(test_path): os.makedirs(test_path)
        if not os.path.exists(test_path_transfer): os.makedirs(test_path_transfer)
            
        for name, files in channels.iteritems():
            # save test channel files
            data_to_file(files, os.path.join(test_path, 'test' + name + '.cfg'))
            data_to_file(files, os.path.join(test_path_transfer, 'test' + name + '.cfg'))

        # save names of predictions
        names = ['pred_ISLES2017_' + os.path.basename(x).split('.')[-3] for x in channels['Channels_ADC']]
        data_to_file(names, os.path.join(test_path, 'testNamesOfPredictions.cfg'))
        data_to_file(names, os.path.join(test_path_transfer, 'testNamesOfPredictions.cfg'))
        
    else:
        
        # set paths for storing channel config files
        train_path = os.path.join(s_path, 'configFiles/train')
        train_path_transfer = os.path.join(s_path_transfer, 'configFiles/train')
        validation_path = os.path.join(s_path, 'configFiles/validation')
        validation_path_transfer = os.path.join(s_path_transfer, 'configFiles/validation')

        if not os.path.exists(train_path): os.makedirs(train_path)
        if not os.path.exists(train_path_transfer): os.makedirs(train_path_transfer)
        if not os.path.exists(validation_path): os.makedirs(validation_path)
        if not os.path.exists(validation_path_transfer): os.makedirs(validation_path_transfer)

        # train and validation division point
        train_val_divison = int(np.floor(len(channels['Channels_ADC']) * 0.8))

        # random subject indices
        indices = range(len(channels['Channels_ADC']))
        random.shuffle(indices)
        
        for name, files in channels.iteritems():
            # save train channel files
            data_to_file([files[i] for i in indices[:train_val_divison]], os.path.join(train_path, 'train' + name + '.cfg'))
            data_to_file([files[i] for i in indices[:train_val_divison]], os.path.join(train_path_transfer, 'train' + name + '.cfg'))
            # save validation channel files
            data_to_file([files[i] for i in indices[train_val_divison:]], os.path.join(validation_path, 'validation' + name + '.cfg'))
            data_to_file([files[i] for i in indices[train_val_divison:]], os.path.join(validation_path_transfer, 'validation' + name + '.cfg'))


        # save names of predictions
        names = ['SMIR.ischleseg.' + os.path.basename(x).split('.')[-3]
                 for x in [channels['Channels_MTT'][i] for i in indices[train_val_divison:]]]
        data_to_file(names, os.path.join(validation_path, 'validationNamesOfPredictions.cfg'))
        data_to_file(names, os.path.join(validation_path_transfer, 'validationNamesOfPredictions.cfg'))

# modelConfig,cfg, trainConfig.cfg and testConfig.cfg must be added and modified manually.


